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

class ImagingChannel{
    /**
     * 
     * @param {string} name 
     * @param {string} handle 
     * @param {number} illum_perc 
     * @param {number} exposure_time_ms 
     * @param {number} analog_gain 
     * @param {number} z_offset_um 
     * @param {boolean} enabled
     */
    constructor(name,handle,illum_perc,exposure_time_ms,analog_gain,z_offset_um,enabled=true){
        this.name=name
        this.handle=handle
        this.illum_perc=illum_perc
        this.exposure_time_ms=exposure_time_ms
        this.analog_gain=analog_gain
        this.z_offset_um=z_offset_um
        this.enabled=enabled
    }
}
function microscopeConfigGetDefault(){
    return {
        project_name:"",
        plate_name:"",
        cell_line:"",
    
        // TODO make this configurable somewhere in the GUI
        autofocus_enabled:true,
    
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
            },
    
            /** @type {SiteSelectionCell[]} */
            mask:[]
        },
        
        // some [arbitrary] default
        wellplate_type:"revvity-phenoplate-384",
    
        /** @type{WellIndex[]} */
        plate_wells:[],
    
        /** @type{ImagingChannel[]} */
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
    }
}
const microscope_config=_p.manage(microscopeConfigGetDefault())
// @ts-ignore
function microscopeConfigOverride(new_config){
    // basic fields
    microscope_config.project_name = new_config.project_name
    microscope_config.plate_name = new_config.plate_name
    microscope_config.cell_line = new_config.cell_line

    // grid
    microscope_config.grid.num_x = parseInt(new_config.grid.num_x)
    microscope_config.grid.delta_x_mm = new_config.grid.delta_x_mm
    microscope_config.grid.num_y = parseInt(new_config.grid.num_y)
    microscope_config.grid.delta_y_mm = new_config.grid.delta_y_mm
    microscope_config.grid.num_z = parseInt(new_config.grid.num_z)
    microscope_config.grid.delta_z_um = new_config.grid.delta_z_um
    microscope_config.grid.num_t = parseInt(new_config.grid.num_t)
    microscope_config.grid.delta_t = new_config.grid.delta_t
    microscope_config.grid.mask.length=0
    if(new_config.grid.mask.length>0){
        // @ts-ignore
        microscope_config.grid.mask.splice(0,0,...new_config.grid.mask.map((cell)=>{
            return new SiteSelectionCell(cell.row,cell.col,cell.plane,cell.selected)
        }))
    }

    // wellplate type
    microscope_config.wellplate_type = new_config.wellplate_type

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
        if(!currentPlateType) throw new Error("could not find wellplate type "+microscope_config.wellplate_type)

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

        /** @type{Map<number,WellIndex>} */
        const currentWellLookup=new Map()
        for(let well of microscope_config.plate_wells){
            const index=getIndexFromRowCol(well.row,well.col)
            if(currentWellLookup.has(index)){
                console.error("duplicate well index",index)
                continue
            }
            currentWellLookup.set(index,well)
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
    // catch performance issue where well selection from cache has not finished in reasonable time
    // (if it takes too long, might as well not bother because the user will be very confused)
    setTimeout(function(){
        if(interval_id!=-1){
            console.error("loading selected wells took too long. skipping well selection load from cache.")
            clearInterval(interval_id)
        }
    },1e3)
    
    // channels.. to avoid sync issues, go through each channel and update the values instead of fully replacing the entry
    for(let new_channel of new_config.channels){
        for(let channel of microscope_config.channels){
            if(new_channel.handle==channel.handle){
                channel.illum_perc=new_channel.illum_perc
                channel.exposure_time_ms=new_channel.exposure_time_ms
                channel.analog_gain=new_channel.analog_gain
                channel.z_offset_um=new_channel.z_offset_um
                channel.enabled=new_channel.enabled
                break
            }
        }
    }

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

setInterval(function(){
    // store microscope_config in local storage to reload on page refresh
    localStorage.setItem("microscope_config",JSON.stringify(_p.getUnmanaged(microscope_config)))
},1e3) // only every once in a while

// keep track of information sent from the microscope to indicate what status it is in
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
    latest_channel_images:{}
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
        if((view_latest_image instanceof HTMLImageElement) && data.latest_imgs!=null && Object.keys(data.latest_imgs).length>0){
            const img_list=Object.keys(data.latest_imgs).map(k=>data.latest_imgs[k])
            let latest_timestamp=img_list.filter(d=>d!=null).map(d=>d.timestamp).reduce((a,b)=>Math.max(a,b),0)
            let latest_image=img_list.find(d=>d.timestamp==latest_timestamp)

            for(let channel of microscope_config.channels){
                let latest_channel_info=data.latest_imgs[channel.handle]
                if(latest_channel_info==null)continue

                // update async to avoid blocking main thread on image update
                setTimeout(function(){
                    // @ts-ignore
                    microscope_state.latest_channel_images[channel.handle]=latest_channel_info
                },0)
            }

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
                const new_src=src_api_action+"?img_handle="+latest_image.handle
                
                // if the src is a new one
                const new_src_url=new URL(new_src,window.location.origin)
                if(view_latest_image.src!==new_src_url.toString()){
                    // init load (by setting .src), and indicate that loading has not yet finished
                    element_load_state.loaded=false
                    view_latest_image.src=new_src

                    if(latest_image.width_px!=null && latest_image.width_px!=view_latest_image.getAttribute("width"))
                        view_latest_image.setAttribute("width",latest_image.width_px)
                    if(latest_image.height_px!=null && latest_image.height_px!=view_latest_image.getAttribute("height"))
                        view_latest_image.setAttribute("height",latest_image.height_px)

                    let histogram_update_in_progress=false

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
                    //update histogram (async)
                    const histogram_query_data={
                        img_handle:latest_image.handle
                    }
                    if(!histogram_update_in_progress){
                        histogram_update_in_progress=true
                        new XHR(true)
                            .onload(function(xhr){
                                let data=JSON.parse(xhr.responseText)
                                if(data.status!="success"){
                                    console.error("error getting histogram",data)
                                    histogram_update_in_progress=false
                                    return
                                }
                                let hist_data=data.hist_values

                                const trace_data={
                                    x:range(hist_data.length),
                                    // @ts-ignore transform scale to be able to display 0 values on log scale
                                    y:hist_data.map(v=>v+1),
                                    // @ts-ignore display original value (i.e. zero to whatever)
                                    text:hist_data.map(v=>"count: "+v),
                                    type:"scatter",
                                    //orientation:"horizontal",
                                    name:data.channel_name
                                }

                                // @ts-ignore
                                //console.log(Plotly.validate([trace_data]),trace_data)
                                if(plt_num_traces==0){
                                    // @ts-ignore
                                    Plotly.addTraces(histogram_plot_element_id,trace_data,plt_num_traces)

                                    plt_num_traces+=1
                                }else{
                                    // trace update must have a _list_ of x and y values
                                    const trace_update={
                                        x:[trace_data.x],
                                        y:[trace_data.y],
                                        text:[trace_data.text],
                                        name:[trace_data.name]
                                    }
                                    // @ts-ignore
                                    Plotly.restyle(histogram_plot_element_id,trace_update,[0])
                                }

                                histogram_update_in_progress=false
                            })
                            .onerror(function(){
                                console.error("error getting histogram")
                                histogram_update_in_progress=false
                            })
                            .send("/api/action/get_histogram_by_handle",histogram_query_data,"POST")
                    }
                    // if the image is not done loading after 3 seconds, assume it failed
                    loadTimer=setTimeout(f,3e3)

                    // if image is already loaded, call the function immediately
                    if(view_latest_image.complete){
                        f()
                    }
                }
            }
            microscope_state.last_image_channel_name=latest_image.channel.name
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

/**
 * technical specification of a micro well plate type
 */
class WellplateType{
    /**
     * data pulled straight from the database
     * @param {object&{
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

    get num_wells(){
        return this.Num_wells_x*this.Num_wells_y
    }

    /** get all plate specs from the server, and cache the result */
    static all_raw_plates=new XHR(false)
        .onload(function(xhr){
            return JSON.parse(xhr.responseText).wellplate_types
        })
        .send("/api/get_features/hardware_capabilities")

    static get all(){
        let plate_types=WellplateType.all_raw_plates

        /** @type {{name:string,num_wells:number,entries:WellplateType[]}[]} */
        let ret=[]
        for(let plate_type_spec of plate_types){
            const new_plate_type=new WellplateType(plate_type_spec)
            
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
        return ret
    }

    /**
     * 
     * @param {string} handle 
     * @returns {WellplateType?}
     */
    static fromHandle(handle){
        for(let entry of this.all){
            for(let type of entry.entries){
                if(type.Model_id==handle){
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
 * return current machine config state, including initial state sent from microscope
 * (does not include other properties of the microscope, like stage position)
 * @returns {HardwareConfigItem[]}
 */
function getConfigState(){
    //@ts-ignore
    return _p.getUnmanaged(machine_defaults)
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
            throw new Error("command already running")
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
