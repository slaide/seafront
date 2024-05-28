import json, time, os
from flask import Flask, send_from_directory, request

from seaconfig import *
from .hardware.camera import Camera, gxiapi
from .hardware.microcontroller import Microcontroller
Command=Microcontroller.Command

app = Flask(__name__, static_folder='src')

@app.route('/')
def index():
    # send local file "index.html" as response
    return send_from_directory('.', 'index.html')

@app.route(rule='/img/<path:path>')
def send_img_from_local_path(path):
    return send_from_directory('img', path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

@app.route('/src/<path:path>')
def send_js(path):
    return send_from_directory('src', path)

@app.route('/p/<path:path>')
def send_p(path):
    return send_from_directory('p', path)

# dummy data
remote_image_path_map:dict[str,dict]={
    "/server/data/acquisition/a65914/D05_2_3_1_1_fluo405.tiff":{
        "dir":"img",
        "filename":"cells.webp"
    },
    "/server/data/acquisition/a65914/D05_3_5_1_1_fluo405.tiff":{
        "dir":"img",
        "filename":"cells2.webp"
    },
}

@app.route("/api/img_remote")
def send_img_from_remote_storage():
    """
    send an image from remote storage
    """

    # get query parameters
    image_path=request.args.get("image_path")
    if image_path is None:
        return json.dumps({"status":"error","message":"no image_path provided"})

    if image_path not in remote_image_path_map:
        return json.dumps({"status":"error","message":"image_path not found"})

    return send_from_directory(remote_image_path_map[image_path]["dir"], remote_image_path_map[image_path]["filename"])

somemap:dict[str,dict]={}

# get objectives
@app.route("/api/get_features/hardware_capabilities", methods=["GET","POST"])
def get_hardware_capabilities():
    """
    get a list of all the hardware capabilities of the system

    these are the high-level configuration options that the user can select from
    """

    ret={
        "wellplate_types":[
            {
                "name":"Perkin Elmer 96",
                "handle":"pe96",
                "num_rows":8,
                "num_cols":12,
            },
            {
                "name":"Falcon 96",
                "handle":"fa96",
                "num_rows":8,
                "num_cols":12,
            },
            {
                "name":"Perkin Elmer 384",
                "handle":"pe384",
                "num_rows":16,
                "num_cols":24,
            },
            {
                "name":"Falcon 384",
                "handle":"fa384",
                "num_rows":16,
                "num_cols":24,
            },
            {
                "name":"Thermo Fischer 384",
                "handle":"tf384",
                "num_rows":16,
                "num_cols":24,
            }
        ],
        "main_camera_imaging_channels":[c.to_dict() for c in [
            AcquisitionChannelConfig(
                name="Fluo 405 nm Ex",
                handle="fluo405",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="Fluo 488 nm Ex",
                handle="fluo488",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="Fluo 561 nm Ex",
                handle="fluo561",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="Fluo 688 nm Ex",
                handle="fluo688",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="Fluo 730 nm Ex",
                handle="fluo730",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="BF LED Full",
                handle="bfledfull",
                illum_perc=20,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="BF LED Right Half",
                handle="bfledright",
                illum_perc=20,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="BF LED Left Half",
                handle="bfledleft",
                illum_perc=20,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
        ]]
    }
    return json.dumps(ret)

from .config.basics import _get_machine_defaults

@app.route("/api/get_features/machine_defaults", methods=["GET","POST"])
def get_machine_defaults():
    """
    get a list of all the low level machine settings (api wrapper to return json-as-string)

    these settings may be changed on the client side, for individual acquisitions
    (though clearly, this is highly advanced stuff, and may cause irreperable hardware damage!)
    """

    return json.dumps(_get_machine_defaults())

plate=Wellplate(
    Manufacturer="perkin elmer",
    Model_name="384 well plate",
    Model_id_manufacturer="pe-384",
    Model_id="",
    Offset_A1_x_mm=12.1,
    Offset_A1_y_mm=9.0,
    Offset_bottom_mm=14.35-10.4,
    Well_distance_x_mm=4.5,
    Well_distance_y_mm=4.5,
    Well_size_x_mm=3.65,
    Well_size_y_mm=3.65,
    Num_wells_x=24,
    Num_wells_y=16,
)

class Core:
    def __init__(self):
        self.microcontrollers=Microcontroller.get_all()
        self.mc=self.microcontrollers[0]
        self.cams=Camera.get_all()

        abort_startup=False
        if len(self.microcontrollers)==0:
            print("error - no microcontrollers found.")
            abort_startup=True
        if len(self.cams)<2:
            print(f"error - found less than two cameras (found {len(self.cams)})")
            abort_startup=True

        if abort_startup:
            raise RuntimeError("did not find microscope hardware")

        print(f"found {len(self.microcontrollers)} microcontrollers")
        for mc in self.microcontrollers:
            mc.open()

            # if flask server is restarted, we can skip this initialization (the microscope retains its state across reconnects)
            if os.environ.get("WERKZEUG_RUN_MAIN") != "true":

                # reset the MCU
                mc.send_cmd(Command.reset())

                # reinitialize motor drivers and DAC
                mc.send_cmd(Command.initialize())
                mc.send_cmd(Command.configure_actuators())

                # when starting up the microscope, the initial position is considered (0,0,0)
                # even homing considers the limits, so before homing, we need to disable the limits
                mc.send_cmd(Command.set_limit_mm("z",-10.0,"lower"))
                mc.send_cmd(Command.set_limit_mm("z",10.0,"upper"))

                # move objective out of the way
                mc.send_cmd(Command.home("z"))
                # set z limit to (or below) 6.7mm, because above that, the motor can get stuck
                mc.send_cmd(Command.set_limit_mm("z",0.0,"lower"))
                mc.send_cmd(Command.set_limit_mm("z",6.7,"upper"))
                # home x to set x reference
                mc.send_cmd(Command.home("x"))
                # clear clamp in x
                mc.send_cmd(Command.move_by_mm("x",30))
                # then move in position to properly apply clamp
                mc.send_cmd(Command.home("y"))
                # home x again to engage clamp
                mc.send_cmd(Command.home("x"))

                # move to an arbitrary position to disengage the clamp
                mc.send_cmd(Command.move_by_mm("x",30))
                mc.send_cmd(Command.move_by_mm("y",30))

                # and move objective up, slightly
                mc.send_cmd(Command.move_by_mm("z",1))

        print(f"found {len(self.cams)} cameras")
        for i,cam in enumerate(self.cams):
            print(f"camera {i}:")
            print(f" - vendor: {cam.vendor_name}")
            print(f" - model: {cam.model_name}")
            print(f" - sn: {cam.sn}")

            cam.open()

            config=AcquisitionChannelConfig("whatever","dontcare",100,5.0,0,0)
            cam.acquire_with_config(config)

            class Container:
                def __init__(self,total_num_images:int):
                    self.num_images=0
                    self.total_num_images=total_num_images

                    self.total_t=0
                    self.last_imaging_time=time.time()
                def __call__(self,img:gxiapi.RawImage):
                    self.num_images+=1
                    current_time=time.time()
                    delta_t=current_time-self.last_imaging_time
                    if self.num_images>1 and self.total_num_images>1:
                        self.total_t+=delta_t/(self.total_num_images-1)
                    self.last_imaging_time=current_time

                    should_break=self.num_images>self.total_num_images
                    if should_break and self.total_num_images>1:
                        print(f"continuous: overhead at {config.exposure_time_ms}ms exposure time is {self.total_t*1e3-config.exposure_time_ms}ms")
                    return should_break
                
            cam.acquire_with_config(config,mode="until_stop",callback=Container(1))

            cam.close()

        # register url rules requiring machine interaction
        app.add_url_rule(f"/api/get_info/stage_position", f"get_stage_position", self.get_stage_position,methods=["GET","POST"])
        app.add_url_rule(f"/api/action/move_x_by", f"action_move_x_by", self.action_move_x_by,methods=["POST"])
        # register url for start_acquisition
        app.add_url_rule(f"/api/acquisition/start", f"start_acquisition", self.start_acquisition,methods=["POST"])
        # for get_acquisition_status
        app.add_url_rule(f"/api/acquisition/status", f"get_acquisition_status", self.get_acquisition_status,methods=["GET","POST"])
        # for move_to_well
        app.add_url_rule(f"/api/action/move_to_well", f"move_to_well", self.move_to_well,methods=["POST"])

    def move_to_well(self):
        # get well_name from request
        json_data=None
        try:
            json_data=request.get_json()
        except Exception as e:
            pass

        if json_data is None:
            return json.dumps({"status": "error", "message": "no json data"})
        
        well_name=json_data['well_name']

        x_mm=plate.get_well_offset_x(well_name)
        y_mm=plate.get_well_offset_y(well_name)
        print(f"well {well_name} is at",x_mm,y_mm)

        self.mc.send_cmd(Command.move_to_mm("x",x_mm))
        self.mc.send_cmd(Command.move_to_mm("y",y_mm))

        return json.dumps({"status":"success","message":"moved to well "+well_name})

    def get_stage_position(self):
        packet=self.mc.get_packet()
        ret={
            "x_pos_mm":packet.x_pos_mm,
            "y_pos_mm":packet.y_pos_mm,
            "z_pos_mm":packet.z_pos_mm,
        }
        return json.dumps({"status":"success","position":ret})

    def action_move_x_by(self):
        json_data=None
        try:
            json_data=request.get_json()
        except Exception as e:
            pass

        if json_data is None:
            return json.dumps({"status": "error", "message": "no json data"})
        
        distance_mm=json_data['dist_mm']
        
        self.mc.send_cmd(Command.move_by_mm("x",distance_mm))

        return json.dumps({"status": "success","moved_by_x_mm":distance_mm})

    def start_acquisition(self):
        """
        start an acquisition with the given configuration
        """

        # get post data with key "config_file"
        json_data=None
        try:
            json_data=request.get_json()
        except Exception as e:
            pass

        if json_data is None:
            return json.dumps({"status": "error", "message": "no json data"})
        
        config = AcquisitionConfig.from_json(json_data["config_file"])
        print("starting acquisition with config:",config)

        # TODO generate some unique acqisition id to identify this acquisition by
        # must be robust against server restarts, and async requests
        # must also cache previous run results, to allow fetching information from past acquisitions
        acquisition_id="a65914"

        somemap[acquisition_id]={
            "acquisition_id":acquisition_id,
            "acquisition_config":config,
            "num_images_acquired":0,
            "num_images_total":1000,
        }

        return json.dumps({"status": "success","acquisition_id":acquisition_id})

    def get_acquisition_status(self):
        """
        get status of an acquisition
        """

        acquisition_id=None
        try:
            acquisition_id=request.get_json()["acquisition_id"]
        except Exception as e:
            pass

        if acquisition_id is None:
            return json.dumps({"status":"error","message":"no acquisition_id provided"})
        
        # TODO get status of acquisition with id "acquisition_id"
        # just mock something for now

        if acquisition_id not in somemap:
            return json.dumps({"status":"error","message":"acquisition_id is invalid"})

        acq_res=somemap[acquisition_id]
        acq_res["num_images_acquired"]+=1
        if acq_res["num_images_acquired"]>=acq_res["num_images_total"]:
            acq_res["num_images_acquired"]=acq_res["num_images_total"]

        ret={
            # request success
            "status":"success",

            "acquisition_id":acquisition_id,
            "acquisition_status":"running",
            "acquisition_progress":{
                # measureable progress
                "current_num_images":acq_res["num_images_acquired"],
                "time_since_start_s":50,
                "start_time_iso":"2021-01-01T12:00:00",
                "current_storage_usage_GB":4.3,

                # estimated completion time information
                # estimation may be more complex than linear interpolation, hence done on server side
                "estimated_total_time_s":100,

                # last image that was acquired
                "last_image":{
                    "well":"D05",
                    "site":{
                        "x":2,
                        "y":3,
                        "z":1,
                    },
                    "timepoint":1,
                    "channel":"fluo405",
                    "full_path":"/server/data/acquisition/a65914/D05_2_3_1_1_fluo405.tiff"
                },
            },

            # some meta information about the acquisition, derived from configuration file
            "acquisition_meta_information":{
                "total_num_images":acq_res["num_images_total"],
                "max_storage_size_images_GB":10,
            },

            "acquisition_config":{
                # full acquisition config copy here
            },

            "message":f"Acquisition is running, {100*acq_res['num_images_acquired']/acq_res['num_images_total']}% complete"
        }
        if acq_res["num_images_acquired"]==acq_res["num_images_total"]:
            ret["acquisition_status"]="finished"
            ret["message"]="Acquisition is finished"

        return json.dumps(ret)
    
    def close(self):
        for mc in self.microcontrollers:
            mc.close()



if __name__ == "__main__":
    core=Core()
    
    app.run(debug=True, port=5002)

    core.close()
