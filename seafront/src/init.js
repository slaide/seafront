/** from p2.js */
let _p=new Manager()

new XHR(false)
    .onload(function(xhr){
        let data=JSON.parse(xhr.responseText)
        for(let entry of data){
            if(entry.name=="microscope name"){
                console.log("connected to microscope: ",entry.value)
                document.title=entry.value
            }
        }
    })
    .send("/api/get_features/machine_defaults")

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
            let channels=JSON.parse(xhr.responseText).main_camera_imaging_channels.map(channel=>{
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

let microscope_state=_p.manage({
    pos:{
        x_mm:12.123,
        y_mm:23.123,
        z_um:4312.0,
    },
})
let updateInProgress=false
function updateMicroscopePosition(){
    if(updateInProgress)return;

    updateInProgress=true
    new XHR(true)
        .onload(function(xhr){
            let data=JSON.parse(xhr.responseText)

            microscope_state.pos.x_mm=data.position.x_pos_mm
            microscope_state.pos.y_mm=data.position.y_pos_mm
            microscope_state.pos.z_um=data.position.z_pos_mm*1e3
        
            updateInProgress=false
        })
        .onerror(function(){
            console.error("error updating microscope position")
        
            updateInProgress=false
        })
        .send("/api/get_info/stage_position")
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
