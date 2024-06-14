/** from p2.js */
let _p=new Manager()

class ImagingChannel{
    /**
     * 
     * @param {string} name 
     * @param {string} handle 
     * @param {number} illum_perc 
     * @param {number} exposure_time_ms 
     * @param {number} analog_gain 
     * @param {number} z_offset_um 
     */
    constructor(name,handle,illum_perc,exposure_time_ms,analog_gain,z_offset_um){
        this.name=name
        this.handle=handle
        this.illum_perc=illum_perc
        this.exposure_time_ms=exposure_time_ms
        this.analog_gain=analog_gain
        this.z_offset_um=z_offset_um
    }
}

let microscope_config=_p.manage({
    project_name:"",
    plate_name:"",
    cell_line:"",

    grid:{
        num_x:2,
        delta_x_mm:0.9,

        num_y:2,
        delta_y_mm:0.9,

        num_z:1,
        delta_z_um:5,

        num_t:1,
        delta_t:{
            h:2,
            m:1,
            s:4,
        }
    },
    
    wellplate_type:"fa96",

    channels:new XHR(false)
        .onload(function(xhr){
            let data=JSON.parse(xhr.responseText)
            let channels=data.main_camera_imaging_channels.map(
                /** @ts-ignore */
                function(channel){
                    return new ImagingChannel(
                        channel.name,
                        channel.handle,
                        channel.illum_perc,
                        channel.exposure_time_ms,
                        channel.analog_gain,
                        channel.z_offset_um
                    )
                }
            )
            return channels
        })
        .onerror(()=>{
            console.error("error getting channels")
        })
        .send("/api/get_features/hardware_capabilities"),
})

// 
let microscope_state=_p.manage({
    machine_name:"",
    state:"idle",
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
    last_image_channel_name:"No image acquired yet",
})
let last_update_successful=true
let updateInProgress=false
/** @type{number?} */
let loadTimer=null
/** @type{Map<HTMLImageElement,object&{loaded:boolean}>} */
const element_image_load_state=new Map()
function updateMicroscopePosition(){
    if(updateInProgress)return;

    updateInProgress=true

    function onerror(){
        if(last_update_successful){
            console.error("error updating microscope position")
            last_update_successful=false
        }
    
        updateInProgress=false
    }
    /** @param {XMLHttpRequest} xhr  */
    function onload(xhr){
        let data=JSON.parse(xhr.responseText)
        
        if(!data.stage_position){
            return onerror()
        }

        if(!last_update_successful){
            console.log("recovered microscope position update")
            last_update_successful=true
        }

        microscope_state.pos.x_mm=data.stage_position.x_pos_mm
        microscope_state.pos.y_mm=data.stage_position.y_pos_mm
        microscope_state.pos.z_um=data.stage_position.z_pos_mm*1e3

        microscope_state.state=data.state

        let view_latest_image=document.getElementById("view_latest_image")
        if((view_latest_image instanceof HTMLImageElement) && data.latest_img!=null){
            if(!element_image_load_state.has(view_latest_image)){
                // indicate that the current state has finished loading on initilization
                element_image_load_state.set(view_latest_image,{loaded:true})
            }

            // check if the image has finished loading
            let element_load_state=element_image_load_state.get(view_latest_image)
            if(!element_load_state){throw new Error("element load state is null")}
            if(!element_load_state.loaded){
                // skip updating if loading is still ongoing, otherwise processing will stop and
                // nothing will be displayed while a new image arrives
                console.warn("image not loaded yet, skipping update")
            }else{
                let src_api_action=""
                if(microscope_state.streaming){
                    src_api_action="/img/get_by_handle_preview"
                }else{
                    src_api_action="/img/get_by_handle"
                }
                const new_src=src_api_action+"?img_handle="+data.latest_img.handle
                
                // if the src is a new one
                if(view_latest_image.src!==new_src){
                    // indicate that loading has not yet finished, init load (by setting .src)
                    element_load_state.loaded=false
                    view_latest_image.src=new_src

                    if(data.latest_img.width_px!=null && data.latest_img.width_px!=view_latest_image.getAttribute("width"))
                        view_latest_image.setAttribute("width",data.latest_img.width_px)
                    if(data.latest_img.height_px!=null && data.latest_img.height_px!=view_latest_image.getAttribute("height"))
                        view_latest_image.setAttribute("height",data.latest_img.height_px)

                    // set callback on load finish
                    const f=function(){
                        // remove the callback
                        if(loadTimer!=null){
                            clearTimeout(loadTimer)
                            loadTimer=null
                        }
                        view_latest_image.removeEventListener("load",f)
                        view_latest_image.removeEventListener("error",f)

                        element_load_state.loaded=true
                    }
                    // consider image loaded either on actual load, or on error
                    view_latest_image.addEventListener("load",f,{once:true})
                    view_latest_image.addEventListener("error",f,{once:true})
                    // if the image is not done loading after 3 seconds, assume it failed
                    loadTimer=setTimeout(f,3e3)

                    // if image is already loaded, call the function immediately
                    if(view_latest_image.complete){
                        f()
                    }
                }
            }
            microscope_state.last_image_channel_name=data.latest_img.channel.name
        }
    
        updateInProgress=false
    }
    new XHR(true)
        .onload(onload)
        .onerror(onerror)
        .send("/api/get_info/current_state")
}
setInterval(updateMicroscopePosition,1e3/15)

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

class WellplateType{
    /**
     * 
     * @param {string} handle 
     * @param {string} name 
     */
    constructor(handle,name){
        this.handle=handle
        this.name=name
    }

    get num_cols(){
        if(this.handle.endsWith("96")){
            return 12
        }else if(this.handle.endsWith("384")){
            return 24
        }else{
            throw new Error("unknown plate type "+this.handle+" for well navigator")
        }
    }
    get num_rows(){
        if(this.handle.endsWith("96")){
            return 8
        }else if(this.handle.endsWith("384")){
            return 16
        }else{
            throw new Error("unknown plate type "+this.handle+" for well navigator")
        }
    }

    static get all(){
        let plate_types=new XHR(false)
            .onload(function(xhr){
                return JSON.parse(xhr.responseText).wellplate_types
            })
            .send("/api/get_features/hardware_capabilities")

        /** @type {{name:string,num_wells:number,entries:WellplateType[]}[]} */
        let ret=[]
        for(let plate_type of plate_types){
            let num_wells=plate_type.num_cols*plate_type.num_rows
            /** @type {WellplateType[]?} */
            let entries=null
            for(let e of ret){
                if(e.num_wells==num_wells){
                    entries=e.entries
                    break
                }
            }
            if(entries==null){
                entries=[]
                ret.push({name:num_wells+" Well Plates",num_wells:num_wells,entries:entries})
            }
            entries.push(new WellplateType(plate_type.handle,plate_type.name))
        }
        return ret
    }

    /**
     * 
     * @param {string} handle 
     * @returns {null|WellplateType}
     */
    static fromHandle(handle){
        for(let entry of this.all){
            for(let type of entry.entries){
                if(type.handle==handle){
                    return type
                }
            }
        }
        return null
    }
}
/**
 * @typedef {object&{value_kind:string,name:string,handle:string,value:string|number}} HardwareConfigItem
 */

/**
 * low level machine control parameters, some of which may be configured/changed for an acquisition
 * @type {HardwareConfigItem[]}
 */
let machine_defaults=_p.manage(new XHR(false)
    .onload(function(xhr){
        let data=JSON.parse(xhr.responseText)

        for(let entry of data){
            if(entry.name=="microscope name"){
                console.log("connected to microscope: ",entry.value)
                microscope_state.machine_name=entry.value
            }
        }

        return data
    })
    .send("/api/get_features/machine_defaults"))

/**
 * return current config state, including initial state sent from microscope
 * (does not include other properties of the microscope, like stage position)
 * @returns {object}
 */
function getConfigState(){
    return {
        machine_config:_p.getUnmanaged(machine_defaults)
    }
}