import typing as tp
import pathlib as path
import datetime as dt

import seaconfig as sc
from pydantic import BaseModel,Field
import tifffile

from .commands import *

async def store_image(
    image_entry:ImageStoreEntry,
    img_compression_algorithm:str,

    metadata:tp.Dict[str,str],
):
    """
    store img as .tiff file, with compression and some metadata

    params:
        metadata: embedded into the file
            e.g.:
                {
                    "BitsPerPixel":latest_channel_image.bit_depth, # type:ignore
                    "BitPaddingInfo":"lower bits are padding with 0s",

                    "LightSourceName":channel.name,
                    "PixelSizeUM":f"{PIXEL_SIZE_UM:.3f}",

                    "ExposureTime":f"{int(channel.exposure_time_ms*1e3)}",
                    "ExposureTimeMS":f"{channel.exposure_time_ms:.3f}",
                    "AnalogGainDB":f"{channel.analog_gain:.1f}",

                    "Make":core_main_cam.vendor_name,
                    "Model":core_main_cam.model_name,
                }
    """

    image_storage_path=image_entry.info.storage_path
    assert isinstance(image_storage_path,str), f"{image_entry.info.storage_path} is not str"

    # takes 70-250ms
    tifffile.imwrite(
        image_storage_path,
        image_entry._img,

        compression=img_compression_algorithm,
        compressionargs={},# for zlib: {'level': 8},
        maxworkers=1,

        # this metadata is just embedded as a comment in the file
        metadata=metadata,

        photometric="minisblack", # zero means black
    )

    print(f"stored image to {image_storage_path}")


class ProtocolGenerator(BaseModel):
    """
    microscope independent protocol that generates commands to be executed (generates on the fly, based on results of previous commands, e.g. based on measured z offset)
    params:
        img_compression_algorithm: tp.Literal["LZW","zlib"] - lossless compression algorithms only
    """

    config_file:sc.AcquisitionConfig
    handle_q_in:tp.Callable[[],None]
    plate:sc.Wellplate
    acquisition_status:AcquisitionStatus
    acquisition_id:str
    ongoing_image_store_tasks:list

    img_compression_algorithm:tp.Literal["LZW","zlib"]="LZW"

    # the values below are initialized during post init hook
    num_wells:int=-1
    num_sites:int=-1
    num_channels:int=-1
    num_channel_z_combinations:int=-1
    num_images_total:int=-1
    site_topleft_x_mm:float=-1
    site_topleft_y_mm:float=-1
    image_size_bytes:int=-1
    max_storage_size_images_GB:float=-1
    project_output_path:path.Path=Field(default_factory=path.Path)

    latest_channel_images:tp.Dict[str,ImageStoreEntry]=Field(default_factory=dict)

    @property
    def well_sites(self):
        "selected sites in the grid mask"
        return [s for s in self.config_file.grid.mask if s.selected]
    @property
    def channels(self):
        "selected channels"
        return [c for c in self.config_file.channels if c.enabled]

    # pydantics version of dataclass.__post_init__
    def model_post_init(self,__context):
        self.num_wells=len(self.config_file.plate_wells)
        self.num_sites=len(self.well_sites)
        self.num_channels=len([c for c in self.config_file.channels if c.enabled])
        self.num_channel_z_combinations=sum((c.num_z_planes for c in self.config_file.channels))
        self.num_images_total=self.num_wells*self.num_sites*self.num_channel_z_combinations

        # the grid is centered around the center of the well
        self.site_topleft_x_mm=self.plate.Well_size_x_mm / 2 - ((self.config_file.grid.num_x-1) * self.config_file.grid.delta_x_mm) / 2
        "offset of top left site from top left corner of the well, in x, in mm"
        self.site_topleft_y_mm=self.plate.Well_size_y_mm / 2 - ((self.config_file.grid.num_y-1) * self.config_file.grid.delta_y_mm) / 2
        "offset of top left site from top left corner of the well, in y, in mm"

        g_config=GlobalConfigHandler.get_dict()

        # calculate meta information about acquisition
        cam_img_width=g_config['main_camera_image_width_px']
        assert cam_img_width is not None
        target_width:int=cam_img_width.intvalue
        cam_img_height=g_config['main_camera_image_height_px']
        assert cam_img_height is not None
        target_height:int=cam_img_height.intvalue
        # get byte size per pixel from config main camera pixel format
        main_cam_pix_format=g_config['main_camera_pixel_format']
        assert main_cam_pix_format is not None
        match main_cam_pix_format.value:
            case "mono8":
                bytes_per_pixel=1
            case "mono10":
                bytes_per_pixel=2
            case "mono12":
                bytes_per_pixel=2
            case _unexpected:
                error_internal(detail=f"unexpected main camera pixel format '{_unexpected}' in {main_cam_pix_format}")

        self.image_size_bytes=target_width*target_height*bytes_per_pixel
        self.max_storage_size_images_GB=self.num_images_total*self.image_size_bytes/1024**3

        base_storage_path_item=g_config["base_image_output_dir"]
        assert base_storage_path_item is not None
        assert type(base_storage_path_item.value) is str
        base_storage_path=path.Path(base_storage_path_item.value)
        assert base_storage_path.exists(), f"{base_storage_path = } does not exist"

        self.project_output_path=base_storage_path/self.config_file.project_name/self.config_file.plate_name/self.acquisition_id
        self.project_output_path.mkdir(parents=True)

        # write config file to output directory
        with (self.project_output_path/"config.json").open("w") as file:
            file.write(self.config_file.json())

    def generate(self)->tp.Generator[
        # yielded types: None means done, str is returned on first iter, other types are results of BaseCommands
        tp.Union[None,tp.Literal["ready"],BaseCommand],
        # received types (at runtime must match return type of <yielded type>.run()."resulttype")
        tp.Union[None,tp.Any],
        # generator return value
        None
    ]:
        # print(f"acquiring {num_wells} wells, {num_sites} sites, {num_channels} channels, i.e. {num_images_total} images, taking up to {max_storage_size_images_GB}GB")

        # 
        Z_STACK_COUNTER_BACKLASH_MM=40e-3 # 40um
        PIXEL_SIZE_UM=900/3000 # 3000px wide fov covers 0.9mm
        # movement below this threshold is not performed
        DISPLACEMENT_THRESHOLD_UM: float=0.5

        # counters on acquisition progress
        start_time=time.time()
        start_time_iso_str=sc.datetime2str(dt.datetime.now(dt.timezone.utc))
        last_image_information=None

        num_images_acquired=0
        storage_usage_bytes=0

        g_config=GlobalConfigHandler.get_dict()
        
        # first yield indicates that this generator is ready to produce commands
        # the value from the consumer on the first yield is None
        # i.e. this MUST be the first yield !!
        yield "ready"

        # get current z coordinate as z reference
        last_position=yield MC_getLastPosition()
        assert isinstance(last_position,mc.Position), f"{type(last_position)=}"
        reference_z_mm=last_position.z_pos_mm

        # if laser autofocus is enabled, use autofocus z reference as initial z reference
        gconfig_refzmm_item=g_config["laser_autofocus_calibration_refzmm"]
        if self.config_file.autofocus_enabled and gconfig_refzmm_item is not None:
            reference_z_mm=gconfig_refzmm_item.floatvalue

            res=yield MoveTo(x_mm=None,y_mm=None,z_mm=reference_z_mm)
            assert isinstance(res,BasicSuccessResponse), f"{type(res)=}"

        # run acquisition:

        # for each timepoint, starting at 1
        for timepoint in range(1,self.config_file.grid.num_t+1):
            for well in self.config_file.plate_wells:
                # these are xy sites
                print_time("before next site")
                for site_index,site in enumerate(self.well_sites):
                    self.handle_q_in()

                    # go to site
                    site_x_mm=self.plate.get_well_offset_x(well.well_name) + self.site_topleft_x_mm + site.col * self.config_file.grid.delta_x_mm
                    site_y_mm=self.plate.get_well_offset_y(well.well_name) + self.site_topleft_y_mm + site.row * self.config_file.grid.delta_y_mm

                    res=yield MoveTo(x_mm=site_x_mm,y_mm=site_y_mm)
                    assert isinstance(res,BasicSuccessResponse), f"{type(res)=}"

                    print_time("moved to site")

                    # run autofocus
                    if self.config_file.autofocus_enabled:
                        for autofocus_attempt_num in range(3):
                            current_displacement=yield AutofocusMeasureDisplacement(config_file=self.config_file)
                            assert isinstance(current_displacement,AutofocusMeasureDisplacementResult), f"{type(current_displacement)=}"
                            
                            current_displacement_um=current_displacement.displacement_um
                            assert current_displacement_um is not None
                            # print(f"measured offset of {current_displacement_um:.2f}um on attempt {autofocus_attempt_num}")
                            if np.abs(current_displacement_um)<DISPLACEMENT_THRESHOLD_UM:
                                break
                            
                            res=yield AutofocusApproachTargetDisplacement(target_offset_um=0,config_file=self.config_file,pre_approach_refz=False)
                            assert isinstance(res,BasicSuccessResponse), f"{type(res)=}"

                    # reference for channel z offsets
                    # (this position may have been adjusted by the autofocus system, but even without autofocus the current position must be the reference)
                    last_position=yield MC_getLastPosition()
                    assert isinstance(last_position,mc.Position), f"{type(last_position)=}"
                    reference_z_mm=last_position.z_pos_mm

                    print_time("af performed")
                    
                    # z stack may be different for each channel, hence:
                    # 1. get list of (channel_z_index,channel,z_relative_to_reference), which may contain each channel more than once
                    # 2. order by z, in ascending (low to high)
                    # 3. move to lowest, start imaging while moving up
                    # 4. move to reference z again in preparation for next site

                    image_pos_z_list:list[tuple[int,float,sc.AcquisitionChannelConfig]]=[]
                    for channel in self.channels:
                        channel_delta_z_mm=channel.delta_z_um*1e-3

                        # <channel reference> is <site reference>+<channel z offset>
                        base_z=reference_z_mm+channel.z_offset_um*1e-3

                        # lower z base is <channel reference> adjusted for z stack, where \
                        # n-1 z movements are performed, half of those below, half above <channel ref>
                        base_z-=((channel.num_z_planes-1)/2)*channel_delta_z_mm

                        for i in range(channel.num_z_planes):
                            i_offset_mm=i*channel_delta_z_mm
                            target_z_mm=base_z+i_offset_mm
                            image_pos_z_list.append((i,target_z_mm,channel))

                    # sort in z
                    image_pos_z_list=sorted(image_pos_z_list,key=lambda v:v[1])

                    res=yield MoveTo(x_mm=None,y_mm=None,z_mm=image_pos_z_list[0][1]-Z_STACK_COUNTER_BACKLASH_MM)
                    assert isinstance(res,BasicSuccessResponse), f"{type(res)=}"
                    res=yield MoveTo(x_mm=None,y_mm=None,z_mm=image_pos_z_list[0][1])
                    assert isinstance(res,BasicSuccessResponse), f"{type(res)=}"

                    print_time("moved to init z")

                    for plane_index,channel_z_mm,channel in image_pos_z_list:
                        site_x=site.col
                        site_y=site.row
                        site_z=plane_index

                        self.handle_q_in()

                        # move to channel offset
                        last_position=yield MC_getLastPosition()
                        assert isinstance(last_position,mc.Position), f"{type(last_position)=}"
                        current_z_mm=last_position.z_pos_mm

                        distance_z_to_move_mm=channel_z_mm-current_z_mm
                        if np.abs(distance_z_to_move_mm)>DISPLACEMENT_THRESHOLD_UM:
                            res=yield MoveTo(x_mm=None,y_mm=None,z_mm=channel_z_mm)
                            assert isinstance(res,BasicSuccessResponse), f"{type(res)=}"

                        last_position=yield MC_getLastPosition()
                        assert isinstance(last_position,mc.Position), f"{type(last_position)=}"

                        print_time("moved to channel z")

                        # snap image
                        res=yield ChannelSnapshot(channel=channel)
                        if not isinstance(res,ImageAcquiredResponse):
                            error_internal(detail=f"failed to snap image at site {site} in well {well} (invalid result type {type(res)})")

                        print_time("snapped image")
                        
                        # store image
                        image_storage_path=f"{str(self.project_output_path)}/{well.well_name}_s{site_index}_x{site.col+1}_y{site.row+1}_z{plane_index+1}_{channel.handle}.tiff"

                        image_store_entry=ImageStoreEntry(
                            pixel_format=g_config["main_camera_pixel_format"].strvalue,
                            info=ImageStoreInfo(
                                channel=channel,
                                width_px=res._img.shape[1],
                                height_px=res._img.shape[0],
                                timestamp=time.time(),
                                position=SitePosition(
                                    well_name=well.well_name,
                                    site_x=site_x,
                                    site_y=site_y,
                                    site_z=site_z,
                                    x_offset_mm=0,
                                    y_offset_mm=0,
                                    z_offset_mm=0,
                                    position=last_position.pos
                                ),
                                storage_path=str(image_storage_path),
                            )
                        )
                        image_store_entry._img=res._img.copy()

                        self.latest_channel_images[channel.name]=image_store_entry

                        # improve system responsiveness while compressing and writing to disk
                        image_store_task=store_image(
                            image_entry=image_store_entry,
                            img_compression_algorithm=self.img_compression_algorithm,

                            metadata={
                                "BitsPerPixel":f"{image_store_entry.bit_depth}",
                                "Make":"CameraMaker",
                                "Model":"CameraModel",
                                "LightSourceName":channel.name,
                                "ExposureTimeMS":f"{channel.exposure_time_ms:.2f}",
                                "AnalogGainDB":f"{channel.analog_gain:.2f}",
                                "PixelSizeUM":f"{PIXEL_SIZE_UM:.3f}",
                            },
                        )
                        self.ongoing_image_store_tasks.append(asyncio.create_task(image_store_task))

                        # pop finished tasks off queue (in-place, hence pop()+push())
                        running_tasks=[]
                        num_done=0
                        while len(self.ongoing_image_store_tasks)>0:
                            task=self.ongoing_image_store_tasks.pop()
                            if task.done():
                                num_done+=1
                            else:
                                running_tasks.append(task)
                        print_time(f"image store tasks: {len(running_tasks)} running, {num_done} finished")
                        for task in running_tasks:
                            self.ongoing_image_store_tasks.append(task)

                        print_time("scheduled image store")

                        num_images_acquired+=1
                        # get storage size from filesystem because tiff compression may reduce size below size in memory
                        try:
                            file_size_on_disk=path.Path(image_storage_path).stat().st_size
                            storage_usage_bytes+=file_size_on_disk
                        except:
                            # ignore any errors here, because this is not an essential feature
                            pass

                        # status items
                        last_image_information=image_store_entry.info
                        
                        time_since_start_s=time.time()-start_time
                        if num_images_acquired>0:
                            estimated_total_time_s=time_since_start_s*(self.num_images_total-num_images_acquired)/num_images_acquired
                        else:
                            estimated_total_time_s=None

                        print(f"{num_images_acquired}/{self.num_images_total} images acquired")
                        
                        self.acquisition_status.last_status=AcquisitionStatusOut(
                            acquisition_id=self.acquisition_id,
                            acquisition_status="running",
                            acquisition_progress=AcquisitionProgressStatus(
                                # measureable progress
                                current_num_images=num_images_acquired,
                                time_since_start_s=time_since_start_s,
                                start_time_iso=start_time_iso_str,
                                current_storage_usage_GB=storage_usage_bytes/(1024**3),

                                # estimated completion time information
                                # estimation may be more complex than linear interpolation, hence done on server side
                                estimated_total_time_s=estimated_total_time_s,

                                # last image that was acquired
                                last_image=last_image_information,
                            ),

                            # some meta information about the acquisition, derived from configuration file
                            # i.e. this is not updated during acquisition
                            acquisition_meta_information=AcquisitionMetaInformation(
                                total_num_images=self.num_images_total,
                                max_storage_size_images_GB=self.max_storage_size_images_GB,
                            ),

                            acquisition_config=self.config_file,

                            message=f"Acquisition is {(100*num_images_acquired/self.num_images_total):.2f}% complete"
                        )

        # done -> yield None and return
        yield None
