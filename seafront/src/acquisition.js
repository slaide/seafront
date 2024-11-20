let acquisition_progress=_p.manage({
    text:"no acquisition started",
    time_since_start_s:"",
    estimated_time_remaining_msg:"",
    acquisition_id:null
})

function start_acquisition(){
    let data={
        // acquisition config
        "config_file":_p.getUnmanaged(microscope_config)
    }
    
    let progress_element=document.getElementById("acquisition-progress-bar")
    if(!progress_element)throw new Error("progress_element is null")

    // make a raw copy of data to avoid modifying the original data
    data=JSON.parse(JSON.stringify(data))

    data.config_file.plate_wells=data.config_file.plate_wells
        // remove those that are not selected (also removes the headers)
        .filter(well=>well.selected)
        // adjust row and well to be 0-indexed (0 index in frontend is used to denote headers, on the server 0 is used to denote the first row/col)
        .map(well=>(new WellIndex(well.row-1,well.col-1,well.selected)))

    data.config_file.channels=data.config_file.channels
        .filter(channel=>channel.enabled)

    new XHR(false)
        .onload((xhr)=>{
            const data=JSON.parse(xhr.responseText)
            if(data.status!="success"){
                message_open("error","acquisition start failed because:",data.message)
                return
            }

            acquisition_progress.acquisition_id=data.acquisition_id
            message_open("info","acquisition started with id:",acquisition_progress.acquisition_id," ; response text: ",xhr.responseText)
        })
        .onerror((xhr)=>{
            message_open("error","error starting acquisition: ",xhr.responseText)
        })
        .send("/api/acquisition/start",data,"POST")
}

/**
 * @typedef{{
    x:number,
    y:number,
    z:number,
    }} WellSite
* @typedef {{
    well:string,
    site:WellSite
    timepoint:number,
    channel_name:string,
    full_path:string,
    handle:string,
}} LastImageInformation
* @typedef {{
    current_num_images:number,
    time_since_start_s:number,
    start_time_iso:string,
    current_storage_usage_GB:number,

    estimated_total_time_s:number|null,

    last_image:LastImageInformation
 * }} AcquisitionProgressStatus
 * @typedef {{
    total_num_images:number,
    max_storage_size_images_GB:number,
 * }} AcquisitionMetaInformation
 * @typedef {{
    row:number,
    col:number,
    selected:boolean,
 * }} AcquisitionWellSiteConfigurationSiteSelectionItem
 * @typedef {{
    h:number,
    m:number,
    s:number,
 * }} AcquisitionWellSiteConfigurationDeltaTime
 * @typedef {{
    num_x:number,
    delta_x_mm:number,
    num_y:number,
    delta_y_mm:number,
    num_t:number,
    delta_t:AcquisitionWellSiteConfigurationDeltaTime,

    mask:AcquisitionWellSiteConfigurationSiteSelectionItem[],
 * }} AcquisitionWellSiteConfiguration
 * @typedef {{
    row:number,
    col:number,
    selected:boolean,
 * }} PlateWellConfig
 * @typedef {{
    name:string,
    handle:string,
    info:*
    }} ConfigItemOption
 * @typedef {{
    name:string,
    handle:string,
    value_kind:"number"|"text"|"option"|"action",
    value:number|string,
    frozen:boolean,
    options:(ConfigItemOption[])|null
 * }} ConfigItem
 * @typedef {{
    major:number,
    minor:number,
    patch:number,
 * }} Version
 * @typedef {{
    project_name:string,
    plate_name:string,
    cell_line:string,
    
    grid:AcquisitionWellSiteConfiguration,

    wellplate_type:string,
    plate_wells:PlateWellConfig[],

    channels:AcquisitionChannelConfig[],

    autofocus_enabled:boolean,

    machine_config:(ConfigItem[])?,

    comment:string|null,

    spec_version:Version,

    timestamp:string|null,
 * }} AcquisitionConfig
 * @typedef {{
    status:"success",

    acquisition_id:string,
    acquisition_status:"running"|"cancelled"|"completed"|"crashed",
    acquisition_progress:AcquisitionProgressStatus,

    acquisition_meta_information:AcquisitionMetaInformation,

    acquisition_config:AcquisitionConfig,

    message:string,
 * }} AcquisitionStatusOut
 */

/**
 * @param{string} acq_id
 * @param{{load:(acq_stat:AcquisitionStatusOut)=>void,error:()=>void}?} cb
 */
function get_acquisition_info(acq_id,cb){
    const send_data={"acquisition_id":acq_id}
    
    let acquisition_progress_element=document.getElementById("acquisition-progress-bar")
    if(!acquisition_progress_element)throw new Error("progress_element is null")

    new XHR(false)
        .onload((xhr)=>{
            /** @type{AcquisitionStatusOut} */
            let progress=JSON.parse(xhr.responseText)

            if(cb&&cb.load){
                cb.load(progress)
            }
        })
        .onerror(()=>{
            if(cb&&cb.error){
                cb.error()
            }
        })
        .send("/api/acquisition/status",send_data,"POST")
}

/**@typedef {{status:"success"}} AcquisitionCancelResponse*/

function cancel_acquisition(){
    get_current_state({load:(microscope_status)=>{
        if(!microscope_status.current_acquisition_id){
            message_open("error","acquisition cancel failed because: no acquisition is currently in progress")
            return
        }

        get_acquisition_info(
            microscope_status.current_acquisition_id,
            {
                load:(acq_stat)=>{
                    const send_data={acquisition_id:microscope_status.current_acquisition_id}

                    new XHR()
                        .onload((xhr)=>{})
                        .onerror((xhr)=>{
                            message_open("error","error cancelling acquisition because ",xhr.responseText)
                        })
                        .send("/api/acquisition/cancel",send_data,"POST")
                },
                error:()=>{
                    message_open("error","error cancelling acquisition")
                }
            }
        )
    },error:()=>{
        message_open("error","error cancelling acquisition")
    }})
}