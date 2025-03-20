/** from p2.js */
let _p=new Manager()

/**
 * make array of length n, with elements 0,1,2,...,n-1
 * @param {number} n
 * @returns {number[]}
 */
function range(n){
    return Array.from({length:n},(v,i)=>i)
}

/**
 * @type {{
 * machine_name:string,
 * state:string,
 * is_in_loading_position:boolean,
 * streaming:boolean,
 * pos:{x_mm:number,y_mm:number,z_um:number},
 * fov_size:{x_mm:number,y_mm:number},
 * last_image_name:string,
 * latest_channel_images:Map<string,ImageStoreInfo>
 * }}
*/
let _initialMachineConfigInfo={
    machine_name:"",
    state:"idle",
    is_in_loading_position:false,
    streaming:false,
    pos:{
        x_mm:12.123,
        y_mm:23.123,
        z_um:4312.0,
    },
    fov_size:{
        x_mm:0.9,
        y_mm:0.9,
    },
    last_image_name:"No image acquired yet",
    latest_channel_images:new Map(),
}
// keep track of information sent from the microscope to indicate what status it is in
let microscope_state=_p.manage(_initialMachineConfigInfo)

let image_loading={in_progress:false}
/**@type {{id:string|null,status_successful:boolean,ws:WebSocket|null}} */
let last_acquisition={
    id:null,
    status_successful:true,
    ws:null,
}
let last_update_successful=true
/** @type {number?} */
let loadTimer=null
/** @type {Map<HTMLImageElement,{loaded:boolean}>} */
const element_image_load_state=new Map()

/**
 * @type {Array<{load:function(CoreCurrentState):void,error:function():void}>}
 */
let state_callbacks=[]
/**
 * 
 * @param {{load:function(CoreCurrentState):void,error:function():void}} callbacks 
 */
function get_current_state(callbacks){
    state_callbacks.push(callbacks)
}

/**
 * 
 * @param {{width:number,height:number}} payload 
 * @param {any} image_bytes 
 * @returns {Promise<ImageData>}
 */
function image_data_to_imagedata(payload,image_bytes){
    let worker=new Worker("src/image2imagedata_worker.js")
    return new Promise((resolve, reject) => {
        worker.onmessage = (e) => resolve(e.data);
        worker.onerror = (e) => reject(e);
        worker.postMessage({payload:payload,image_bytes:image_bytes});
    });
}

/**
 * 
 * @param {string} channel_name 
 * @returns {Promise<ImageData>}
 */
function fetch_image(channel_name){
    return new Promise((resolve,reject)=>{
        // use websocket only once, as protocol to transfer arbitrary bytes (image data, uncompressed, no standardized format)
        const websocket_protocol=window.location.protocol=="https:"?"wss:":"ws:"
        const websocket_url=websocket_protocol+"//"+window.location.host+"/ws/get_info/acquired_image"
        const ws=new WebSocket(websocket_url)

        /**
         * 
         * @param {MessageEvent} event 
         */
        function outer_message(event){
            let payload=JSON.parse(event.data)

            ws.binaryType="arraybuffer"
            ws.onmessage=async event=>{
                const imagedata=await image_data_to_imagedata(payload,event.data)

                // close websocket connection, otherwise it will stay open forever
                ws.close()

                resolve(imagedata)
            }
        }

        ws.onopen=()=>ws.send(channel_name)
        ws.onmessage=outer_message
        ws.onerror=err=>{
            console.log("img_ws err:",err)
            reject(err)
        }
    })
}

/** @type {WebSocket?} */
let status_update_websocket=null
let last_image={timestamp:0}
/**@type {Map<string,ImageStoreInfo>} */
class ChannelUpdater{
    constructor(){
        /**@type {Map<string,ImageStoreInfo>} */
        this.latest_channel_images=new Map()
        /**@type {Map<string,{p:Promise<{img:ImageData,info:ImageStoreInfo}>,resolved:boolean}>} */
        this.channel_update_in_progress=new Map()
    }
    /**
     * @param {ImageStoreInfo}channel_data
     * @returns {Promise<{img:ImageData,info:ImageStoreInfo}>}
     */
    schedule(channel_data){
        const current_update=this.channel_update_in_progress.get(channel_data.channel.handle)
        if(current_update && !current_update.resolved){
            return current_update.p
        }

        const new_update=fetch_image(channel_data.channel.handle).then(imagedata=>{
            let handle={img:imagedata,info:channel_data}

            this.latest_channel_images.set(channel_data.channel.handle,handle.info)

            let update_handle=this.channel_update_in_progress.get(channel_data.channel.handle)
            if(!update_handle)throw new Error("")
            update_handle.resolved=true

            return handle
        })
        this.channel_update_in_progress.set(channel_data.channel.handle,{p:new_update,resolved:false})

        return new_update
    }
    /**
     * 
     * @param {string} channel_name 
     */
    clear(channel_name){
        this.latest_channel_images.delete(channel_name)
        this.channel_update_in_progress.delete(channel_name)
    }
}
let channel_updater=new ChannelUpdater()

/**
    * @param {ImageData} img 
    * @returns {Promise<number[]>}
    */
function worker_calculate_histogram(img){
    let worker=new Worker("src/histogram_worker.js")
    return new Promise((resolve, reject) => {
        worker.onmessage = (e) => resolve(e.data);
        worker.onerror = (e) => reject(e);
        worker.postMessage(img);
    });
}

/**@type {Map<string,number>} */
let last_channel_update_timestamps=new Map()
/**
 * initiates the background update loop that handles more than just the position
 * (loop -> updates frequently after the first call)
 */
function updateMicroscopePosition(){
    function onerror(){
        if(last_update_successful){
            message_open("error","error updating microscope position")
            last_update_successful=false
        }

        for(let cb of state_callbacks){
            cb.error()
        }
        state_callbacks=[]
    }

    /** @param {CoreCurrentState} data */
    async function onload(data){
        if(!last_update_successful){
            message_open("info","recovered microscope position update")
            last_update_successful=true
        }

        for(let cb of state_callbacks){
            cb.load(data)
        }
        state_callbacks=[]

        microscope_state.pos.x_mm=data.adapter_state.stage_position.x_pos_mm
        microscope_state.pos.y_mm=data.adapter_state.stage_position.y_pos_mm
        microscope_state.pos.z_um=data.adapter_state.stage_position.z_pos_mm*1e3

        microscope_state.state=data.adapter_state.state
        microscope_state.is_in_loading_position=data.adapter_state.is_in_loading_position

        let view_latest_image=document.getElementById("view_latest_image")
        let channel_names=Object.keys(data.latest_imgs)
        if(data.latest_imgs!=null && channel_names.length>0){
            // default: pick first channel
            /**@type {ImageStoreInfo|null} */
            let latest_channel_info=data.latest_imgs[channel_names[0]]

            // remove histograms for channels that are no longer in the config
            /**@type {number[]}*/const indicesToRemove = [];
            if(!histogram_plot_element){console.error("histogram_plot_element not present");return}
            histogram_plot_element.data.forEach((trace, index) => {
                if (!microscope_config.channels.find(c=>trace.name==c.handle && c.enabled)){
                    indicesToRemove.push(index)

                    channel_updater.clear(trace.name)
                    last_channel_update_timestamps.delete(trace.name)
                }
            });
            if (indicesToRemove.length > 0) {
                Plotly.deleteTraces(histogram_plot_element, indicesToRemove);
            }

            const colors = [
                "#1f77b4", // Blue
                "#ff7f0e", // Orange
                "#2ca02c", // Green
                "#d62728", // Red
                "#9467bd", // Purple
                "#8c564b", // Brown
                "#e377c2", // Pink
                "#7f7f7f", // Gray
                "#bcbd22", // Olive
                "#17becf"  // Cyan
            ]              

            // find channel with newest timestamp
            let timestamp=last_image.timestamp
            for(let [channel_name,channel_index] of Object.keys(data.latest_imgs).map(/**@type {function(string,number):[string,number]}*/(v,i)=>[v,i])){
                const channel_data=data.latest_imgs[channel_name]
                if(!channel_data)continue // unreachable, but type checker is not smart enough

                // if data is for a channel that is not currently selected, skip it
                if (!microscope_config.channels.find(c=>channel_name==c.handle && c.enabled)){
                    continue
                }

                const channel_timestamp=parseFloat(""+(channel_data.timestamp||"0"))

                if(channel_timestamp>timestamp){
                    timestamp=channel_timestamp
                    latest_channel_info=channel_data
                }

                if((last_channel_update_timestamps.get(channel_data.channel.handle)||0) < channel_timestamp){
                    last_channel_update_timestamps.set(channel_data.channel.handle,channel_timestamp)

                    channel_updater.schedule(channel_data).then(async data=>{
                        const latest_channel_image_canvas=document.getElementById("view_latest_image_"+data.info.channel.handle)
                        if(!(latest_channel_image_canvas instanceof HTMLCanvasElement)){
                            console.error("latest_channel_image_canvas",latest_channel_image_canvas,typeof latest_channel_image_canvas)
                            // assume page is not done loading, print error to console and try again later
                        }else{
                            latest_channel_image_canvas.getContext("2d")?.putImageData(data.img,0,0)

                            let histogram_data=null
                            try{
                                const prom=worker_calculate_histogram(data.img)
                                histogram_data=await prom
                            }catch(e){
                                console.error(e)
                                return
                            }
                            let histo=histogram_data

                            // slightly wasteful, but nice to copy to other places
                            if(!histogram_plot_element){console.error("histogram_plot_element not present");return}
                            /**@type {number[]}*/const indicesToRemove = [];
                            histogram_plot_element.data.forEach((trace, index) => {
                                if (trace.name === channel_data.channel.handle) {
                                    indicesToRemove.push(index);
                                }
                            });
                            if (indicesToRemove.length > 0) {
                                Plotly.deleteTraces(histogram_plot_element, indicesToRemove);
                            }

                            Plotly.addTraces(histogram_plot_element,{
                                x:histo.map((val,i,ar)=>i),
                                y:histo,
                                type:"scatter",
                                mode:"lines",
                                name:channel_data.channel.handle,
                                legendrank:channel_index,
                                line:{color:colors[channel_index]},
                            })
                        }
                    })
                }
            }

            if(!latest_channel_info)throw new Error("unreachable")

            // if that timestamp is newer than that of the last image that was displayed, fetch new image and display
            if(last_image.timestamp<timestamp){
                last_image.timestamp=timestamp

                channel_updater.schedule(latest_channel_info).then(data=>{
                    const latest_image_canvas=view_latest_image
                    if(!(latest_image_canvas instanceof HTMLCanvasElement)){
                        console.error("view_latest_image",view_latest_image,typeof view_latest_image)
                        // assume page is not done loading, print error to console and try again later
                    }else{
                        microscope_state.last_image_name=data.info.channel.name
                        latest_image_canvas.getContext("2d")?.putImageData(data.img,0,0)
                    }
                })
            }
        }

        let acquisition_progress_element=document.getElementById("acquisition-progress-bar")
        if(!acquisition_progress_element)/*page not finished loading*/return;//throw new Error("progress_element is null")

        /**@param {number} percent */
        function setAcquisitionProgressPercent(percent){
            if(!acquisition_progress_element)throw new Error("progress_element is null")
            acquisition_progress_element.style.setProperty(
                "--percent-done",
                percent + "%"
            )
        }

        // get latest acquisition id
        if(data.current_acquisition_id!=last_acquisition.id){
            last_acquisition.id=data.current_acquisition_id
        }

        // ensure websocket is running to fetch latest acquisition status
        if(last_acquisition.id!=null){
            if(last_acquisition.ws==null){
                const websocket_protocol=window.location.protocol=="https:"?"wss:":"ws:"
                const websocket_url=websocket_protocol+"//"+window.location.host+"/ws/acquisition/status"
                last_acquisition.ws=new WebSocket(websocket_url)

                function get_status(){
                    // a connection that closed before the animation frame is triggered may leave a stale callback to this function
                    // so we act accordingly
                    if(last_acquisition.ws==null)return

                    if(last_acquisition.id!=null){
                        last_acquisition.ws.send(JSON.stringify({"acquisition_id":last_acquisition.id}))
                    }

                    // when an acquisition has stopped, last_acquisition.id will be null, but then a new acquisition may start later
                    // so we still need to query an update, even if no acquisition is currently running
                    requestAnimationFrame(get_status)
                }
                last_acquisition.ws.onopen=()=>{get_status()}
                last_acquisition.ws.onmessage=(event)=>{
                    let data=JSON.parse(JSON.parse(event.data))

                    display_acquisition_status(data)
                }
                last_acquisition.ws.onerror=(event)=>{
                    if(last_acquisition.status_successful){
                        message_open("error","error getting acquisition progress",event)
                    }
                    last_acquisition.status_successful=false
                }
                last_acquisition.ws.onclose=()=>{
                    last_acquisition.ws=null
                }
            }
        }

        /**
         * 
         * @param {AcquisitionStatusOut} progress 
         */
        function display_acquisition_status(progress){
            if(!acquisition_progress_element)throw new Error("progress_element is null")

            if(!last_acquisition.status_successful){
                message_open("info","acquisition progress updated")
            }
            last_acquisition.status_successful=true

            acquisition_progress.time_since_start_s=""+progress.acquisition_progress.time_since_start_s

            // called estimated_total_time_s but actually contains the _remaining_ time in s
            let remaining_time_s_total=progress.acquisition_progress.estimated_total_time_s

            if(remaining_time_s_total!=null){
                let minutes=remaining_time_s_total%3600
                let hours=(remaining_time_s_total-minutes)/3600
                let seconds=minutes%60
                minutes=(minutes-seconds)/60

                let remaining_time_s=seconds.toFixed(0)
                let remaining_time_m=minutes.toFixed(0)
                let remaining_time_h=hours.toFixed(0)

                /**
                 * @param {string} s
                 * @return {string}
                 */
                function pad_to_two_digits(s){
                    if(s.length<2){
                        return "0"+s
                    }
                    return s
                }

                let time_remain_estimate_msg_string=""
                if(remaining_time_s_total>0){
                    time_remain_estimate_msg_string="done in "
                    if(hours>0){
                        time_remain_estimate_msg_string+=remaining_time_h+"h:"
                    }
                    if(minutes>0){
                        if(hours>0){
                            time_remain_estimate_msg_string+=pad_to_two_digits(remaining_time_m)+"m:"
                        }else{
                            time_remain_estimate_msg_string+=remaining_time_m+"m:"
                        }
                    }
                    if(minutes>0){
                        time_remain_estimate_msg_string+=pad_to_two_digits(remaining_time_s)+"s"
                    }else{
                        time_remain_estimate_msg_string+=remaining_time_s+"s"
                    }
                }
                acquisition_progress.estimated_time_remaining_msg=time_remain_estimate_msg_string

                acquisition_progress.text=progress.acquisition_status+" - "+progress.message
                setAcquisitionProgressPercent(
                    progress.acquisition_progress.current_num_images
                    / progress.acquisition_meta_information.total_num_images
                    * 100
                )
            }
        }
    }

    /**
     * call .close on the return value to close the websocket again
     * @returns {WebSocket}
     */
    if(status_update_websocket==null){
        const websocket_protocol=window.location.protocol=="https:"?"wss:":"ws:"
        const websocket_url=websocket_protocol+"//"+window.location.host+"/ws/get_info/current_state"
        const ws=new WebSocket(websocket_url)

        let last_time=performance.now()
        function request_update(){
            ws.send("get_info/current_state")

            const time_since_last_update_ms=performance.now()-last_time
            if(time_since_last_update_ms>300){
                // print informative message if the update delta time exceeds some arbitrary, but longer than expected, delta
                console.log("time since last update:",time_since_last_update_ms,"ms (this is longer than expected, but may happen occasionally)")
            }
            last_time=performance.now()
        }

        ws.onopen=()=>{
            request_update()
        }
        ws.onmessage=async event=>{
            const data=JSON.parse(JSON.parse(event.data))
            await onload(data)

            requestAnimationFrame(request_update)
        }
        ws.onerror=async err_event=>{
            // likely caused by connection issue, which will result in a disconnect/close, which in turn will cause a reconnect, so we do not have to handle errors specifically
            console.error("ws error:",err_event)
            await onerror()
        }
        ws.onclose=()=>{
            // closed externally, usually caused by server disconnect
            console.log("ws closed")

            // clear stored websocket handle (now stale)
            status_update_websocket=null
            // and schedule reconnect attempt
            requestAnimationFrame(updateMicroscopePosition)
        }

        status_update_websocket=ws
    }
}
updateMicroscopePosition()

class ImagingChannel{
    /**
     * 
     * @param {string} name 
     * @param {string} handle 
     * @param {number} illum_perc 
     * @param {number} exposure_time_ms 
     * @param {number} analog_gain 
     * @param {number} z_offset_um 
     * @param {number} num_z_planes
     * @param {number} delta_z_um
     * @param {boolean} enabled
     */
    constructor(name,handle,illum_perc,exposure_time_ms,analog_gain,z_offset_um,num_z_planes,delta_z_um,enabled=true){
        this.name=name
        this.handle=handle
        this.illum_perc=illum_perc
        this.exposure_time_ms=exposure_time_ms
        this.analog_gain=analog_gain
        this.z_offset_um=z_offset_um
        this.num_z_planes=num_z_planes
        this.delta_z_um=delta_z_um

        this.enabled=enabled
    }

    /**
     * @param {AcquisitionChannelConfig} py_obj
     * @return {ImagingChannel}
     */
    static from_python(py_obj){
        return new ImagingChannel(
            py_obj.name,
            py_obj.handle,
            py_obj.illum_perc,
            py_obj.exposure_time_ms,
            py_obj.analog_gain,
            py_obj.z_offset_um,
            py_obj.num_z_planes,
            py_obj.delta_z_um,
        )
    }
}

/**
 * low level machine control parameters, some of which may be configured/changed for an acquisition
 * @return {ConfigItem[]}
 */
function getMachineDefaults(){
    /** @type {ConfigItem[]}*/
    let ret=[]
    new XHR(false)
        .onload(function(xhr){
            let data=JSON.parse(xhr.responseText)

            for(let entry of data){
                if(entry.name=="microscope name"){
                    console.log("connected to microscope: ",entry.value)
                    microscope_state.machine_name=entry.value
                }
            }

            ret=data
        })
        .send("/api/get_features/machine_defaults",{},"POST")

    return ret
}

/**
 * return current machine config state, including initial state sent from microscope
 * (does not include other properties of the microscope, like stage position)
 * @returns {ConfigItem[]}
 */
function getConfigState(){
    //@ts-ignore
    return _p.getUnmanaged(microscope_config.machine_config)
}

/**
 * @return {HardwareCapabilitiesResponse|null}
 */
function get_hardware_capabilities(){
    return new XHR(false)
        .onload(function (xhr) {
            /** @type {HardwareCapabilitiesResponse} */
            let data = JSON.parse(xhr.responseText)
            return data
        })
        .onerror(() => {
            console.error("error getting hardware_capabilities")
        })
        .send("/api/get_features/hardware_capabilities",{},"POST")
}

function microscopeConfigGetDefault(){
    // TODO fetch this [default] from the server

    /** @type {AcquisitionConfig} */
    let ret={
        project_name: "",
        plate_name: "",
        cell_line: "",

        autofocus_enabled: false,

        grid: {
            num_x: 1,
            delta_x_mm: 0.9,

            num_y: 1,
            delta_y_mm: 0.9,

            num_t: 1,
            delta_t: {
                h: 2,
                m: 1,
                s: 4,
            },

            mask: [{row:0,col:0,selected:true}]
        },

        // some [arbitrary] default
        wellplate_type: "revvity-phenoplate-384",

        plate_wells: [],

        channels: (get_hardware_capabilities()?.main_camera_imaging_channels.map(ImagingChannel.from_python)||[]),

        machine_config: getMachineDefaults(),
        comment: null,
        spec_version: {
            major: 0,
            minor: 0,
            patch: 0
        },
        timestamp: null
    }

    return ret
}
const microscope_config=_p.manage(microscopeConfigGetDefault())

const objective={
    fov_size_x_mm:0.9,
    fov_size_y_mm:0.9,
}

const USER_INTERFACE_LIMITS={
    grid_max_num_x:99,
    grid_max_num_y:99,
}

/**
 * @param {string} handle
 * @returns {ConfigItem|null}
 * */
function getMachineConfig(handle){
    if(!microscope_config.machine_config)return null;

    for(let c of microscope_config.machine_config){
        if(c.handle==handle){
            return c
        }
    }
    return null
}

// @ts-ignore
function microscopeConfigOverride(new_config){
    // basic fields
    if(new_config.project_name!=null)
        microscope_config.project_name = new_config.project_name
    if(new_config.plate_name!=null)
        microscope_config.plate_name = new_config.plate_name
    if(new_config.cell_line!=null)
        microscope_config.cell_line = new_config.cell_line

    // grid
    if(new_config.grid!=null){
        if(new_config.grid.num_x!=null)
            microscope_config.grid.num_x = parseInt(new_config.grid.num_x)
        if(new_config.grid.delta_x_mm!=null)
            microscope_config.grid.delta_x_mm = new_config.grid.delta_x_mm
        if(new_config.grid.num_y!=null)
            microscope_config.grid.num_y = parseInt(new_config.grid.num_y)
        if(new_config.grid.delta_y_mm!=null)
            microscope_config.grid.delta_y_mm = new_config.grid.delta_y_mm
        if(new_config.grid.num_t!=null)
            microscope_config.grid.num_t = parseInt(new_config.grid.num_t)
        if(new_config.grid.delta_t!=null)
            microscope_config.grid.delta_t = new_config.grid.delta_t

        if(new_config.grid.mask!=null){
            microscope_config.grid.mask.length=0
            if(new_config.grid.mask.length>0){
                // @ts-ignore
                microscope_config.grid.mask.splice(0,0,...new_config.grid.mask.map((cell)=>{
                    return new SiteSelectionCell(cell.row,cell.col,cell.selected)
                }))
            }
        }

        if(new_config.machine_config!=null){
            if(microscope_config.machine_config!=null){
                microscope_config.machine_config.length=0
            }else{
                microscope_config.machine_config=[]
            }
            microscope_config.machine_config.splice(0,0,...new_config.machine_config)
        }
        try{filter_results()}catch(e){}
    }

    // wellplate type
    if(new_config.wellplate_type!=null)
        microscope_config.wellplate_type = new_config.wellplate_type

    // selected wells (patrick from the future says: this code is weird. i think it tries to wait for some conditions to be present before it initializes something)
    let interval_id=0
    interval_id=setInterval(function(){
        if(interval_id==-1){
            console.error("interval cleared twice")
            return
        }

        // if well list has not updated to be of expected length, wait longer for update
        if(microscope_config.plate_wells.length!=new_config.plate_wells.length){
            return
        }

        const currentPlateType=WellplateType.fromHandle(microscope_config.wellplate_type)
        if(!currentPlateType){
            console.error("could not find wellplate type "+microscope_config.wellplate_type)
            return
        }

        /**
         * calculate well index hash
         * @param {number} row 
         * @param {number} col 
         * @returns {number}
         */
        function getIndexFromRowCol(row,col){
            if(!currentPlateType) throw new Error("unreachable")
            return row*(currentPlateType.num_wells+3)+col
        }

        /** @type {Map<number,WellIndex>} */
        const currentWellLookup=new Map()
        for(let well of microscope_config.plate_wells){
            const index=getIndexFromRowCol(well.row,well.col)
            if(currentWellLookup.has(index)){
                console.error("duplicate well index",index)
                continue
            }
            currentWellLookup.set(index,WellIndex.from_python(well))
        }

        for(let configWell of new_config.plate_wells){
            let currentWell=currentWellLookup.get(getIndexFromRowCol(configWell.row,configWell.col))
            // this should not happen, but if it does, wait longer for update
            if(currentWell==null){
                return
            }
            
            currentWell.selected=configWell.selected
        }

        clearInterval(interval_id)
        interval_id=-1
    },0.01e3)
        
    // channels.. to avoid sync issues, go through each channel and update the values instead of fully replacing the entry
    if(new_config.channels!=null){
        for(let new_channel of new_config.channels){
            for(let channel of microscope_config.channels){
                if(new_channel.handle==channel.handle){
                    channel.illum_perc=new_channel.illum_perc
                    channel.exposure_time_ms=new_channel.exposure_time_ms
                    channel.analog_gain=new_channel.analog_gain
                    channel.z_offset_um=new_channel.z_offset_um
                    
                    channel.num_z_planes=new_channel.num_z_planes
                    channel.delta_z_um=new_channel.delta_z_um

                    channel.enabled=new_channel.enabled
                    break
                }
            }
        }
    }

    // autofocus enabled
    if(new_config.autofocus_enabled!=null)
        microscope_config.autofocus_enabled=new_config.autofocus_enabled
}
function microscopeConfigReset(){
    microscopeConfigOverride(microscopeConfigGetDefault())
}

setTimeout(function(){
    let existing_config_item=localStorage.getItem("microscope_config")
    if(!existing_config_item)return

    let existing_config=JSON.parse(existing_config_item)

    try{
        microscopeConfigOverride(existing_config)
    }catch(e){
        console.error("error loading microscope config from local storage. resetting.",e)
        microscopeConfigReset()
    }
},0)

function storeConfigIntoBrowserLocalStorage(){
    // store microscope_config in local storage to reload on page refresh
    localStorage.setItem("microscope_config",JSON.stringify(_p.getUnmanaged(microscope_config)))
}
setInterval(storeConfigIntoBrowserLocalStorage,1e3) // once per second

/**
 * list that only contains selected channels (to simplify GUI)
 * @type {ImagingChannel[]}
 */
let selected_channels=_p.manage([])
function initSelectedChannels(){
    for(let channel of microscope_config.channels){
        channel=_p.ensureManagedObject(channel)

        if(channel.enabled){
            // @ts-ignore
            selected_channels.push(channel)
        }

        _p.onValueChangeCallback(()=>channel.enabled,function(_channel_is_now_enabled){
            // regenerate list as easy solution to ensure order of channels is consistent
            // (this has bad performance when multiple elements change in quick succession, but the list is small enough that it doesn't matter)
            selected_channels.length=0
            selected_channels.splice(0,0,...microscope_config.channels.filter(c=>c.enabled))
        },{cache:true})
    }
}
initSelectedChannels()

const limits={
    illumination_percent:{
        min:20,
        max:100,
        default:100,
    },
    exposure_time_ms:{
        min:0.1,
        max:936,
        default:5.0,
    },
    analog_gain:{
        min:0,
        max:24,
        default:0,
    },
    z_offset_um:{
        min:-50,
        max:50,
        default:0,
    }
}

/**
 * technical specification of a micro well plate type
 */
class WellplateType{
    /**
     * data pulled straight from the database
     * @param {{
     *     Manufacturer: string,
     *     Model_name: string,
     *     Model_id: string,
     *     Num_wells_y: number,
     *     Num_wells_x: number,
     *     Length_mm: number,
     *     Width_mm: number,
     *     Well_size_x_mm: number,
     *     Well_size_y_mm: number,
     *     Offset_A1_x_mm: number,
     *     Offset_A1_y_mm: number,
     *     Well_distance_x_mm: number,
     *     Well_distance_y_mm: number,
     *     Offset_bottom_mm: number
     * }} spec json object with several properties
     */
    constructor(spec){
        this.Manufacturer = spec.Manufacturer
        this.Model_name = spec.Model_name
        this.Model_id = spec.Model_id
        this.Num_wells_y = spec.Num_wells_y
        this.Num_wells_x = spec.Num_wells_x
        this.Length_mm = spec.Length_mm
        this.Width_mm = spec.Width_mm
        this.Well_size_x_mm = spec.Well_size_x_mm
        this.Well_size_y_mm = spec.Well_size_y_mm
        this.Offset_A1_x_mm = spec.Offset_A1_x_mm
        this.Offset_A1_y_mm = spec.Offset_A1_y_mm
        this.Well_distance_x_mm = spec.Well_distance_x_mm
        this.Well_distance_y_mm = spec.Well_distance_y_mm
        this.Offset_bottom_mm = spec.Offset_bottom_mm
    }

    /**
     * @param {Wellplate} py_obj
     * @return {WellplateType}
     */
    static from_python(py_obj){
        return new WellplateType(py_obj)
    }

    get num_wells(){
        return this.Num_wells_x*this.Num_wells_y
    }

    /** get all plate specs from the server, and cache the result */
    static all_raw_plates=(get_hardware_capabilities()?.wellplate_types.map(WellplateType.from_python)||[])

    /** @type {{name:string,num_wells:number,entries:WellplateType[]}[]?} */
    static _all=null

    static _handleToType=new Map()

    /** @type {{name:string,num_wells:number,entries:WellplateType[]}[]} */
    static get all(){
        if(WellplateType._all!=null){
            return WellplateType._all
        }

        let plate_types=WellplateType.all_raw_plates

        /** @type {{name:string,num_wells:number,entries:WellplateType[]}[]} */
        let ret=[]
        for(let new_plate_type of plate_types){
            WellplateType._handleToType.set(new_plate_type.Model_id,new_plate_type)
            
            /** @type {WellplateType[]?} */
            let entries=null
            for(let e of ret){
                if(e.num_wells==new_plate_type.num_wells){
                    entries=e.entries
                    break
                }
            }
            if(entries==null){
                entries=[]
                ret.push({name:new_plate_type.num_wells+" well plates",num_wells:new_plate_type.num_wells,entries:entries})
            }
            entries.push(new_plate_type)
        }
        
        WellplateType._all=ret

        return ret
    }

    /**
     * 
     * @param {string} handle 
     * @returns {WellplateType?}
     */
    static fromHandle(handle){
        return WellplateType._handleToType.get(handle)
    }
}

class CommandProgressIndicator{
    constructor(){
        this.current_command=null
    }
    get statusText(){
        if(this.current_command==null){
            return "Idle"
        }
        
        return this.current_command
    }
    /**
     * 
     * @param {string} command 
     */
    run(command){
        if(this.current_command!=null){
            throw new Error("another command is currently running")
        }
        
        this.current_command=command
        // change cursor to indicate command is running
        document.body.classList.add("command-is-running")
    }
    stop(){
        this.current_command=null
        // change cursor back to normal
        document.body.classList.remove("command-is-running")
    }
}

const progress_indicator=new CommandProgressIndicator()
