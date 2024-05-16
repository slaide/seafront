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
    base_path:"/home/pharmbio/Downloads/",
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

    objective:"20xolympus",
    wellplate_type:"fa96",

    main_camera_trigger:"software",
    main_camera_pixel_type:"mono12",

    laser_af_enabled:false,

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
        z_um:4312,
    },
})

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

const main_camera_triggers=new XHR(false)
    .onload(function(xhr){
        let triggers=JSON.parse(xhr.responseText).main_camera_triggers.map(trigger=>{
            return {
                name:trigger.name,
                handle:trigger.handle,
            }
        })
        return triggers
    })
    .send("/api/get_features/hardware_capabilities")

const main_camera_pixel_types=new XHR(false)
    .onload(function(xhr){
        let triggers=JSON.parse(xhr.responseText).main_camera_pixel_formats.map(format=>{
            return {
                name:format.name,
                handle:format.handle,
            }
        })
        return triggers
    })
    .send("/api/get_features/hardware_capabilities")

class Objective{
    /**
     * 
     * @param {string} name 
     * @param {string} handle 
     * @param {number} magnification
     */
    constructor(name,handle,magnification){
        this.name=name
        this.handle=handle
        this.magnification=magnification
    }

    static get all(){
        return new XHR(false)
            .onload(function(xhr){
                let triggers=JSON.parse(xhr.responseText).main_camera_objectives.map(objective=>{
                    return new Objective(
                        objective.name,
                        objective.handle,
                        objective.magnification,
                    )
                })
                return triggers
            })
            .send("/api/get_features/hardware_capabilities")
    }

    /**
     * return name of an objective when given its handle
     * @param {string} handle 
     * @returns {null|Objective} 
     */
    static fromHandle(handle){
        for(let o of Objective.all){
            if(o.handle==handle){
                return o
            }
        }
        return null
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
        return ret;
        [
            {
                name:"96 Well Plate",
                entries:[
                    new WellplateType("pe96","Perkin Elmer 96"),
                    new WellplateType("fa96","Falcon 96"),
                ]
            },
            {
                name:"384 Well Plate",
                entries:[
                    new WellplateType("pe384","Perkin Elmer 384"),
                    new WellplateType("fa384","Falcon 384"),
                    new WellplateType("tf384","Thermo Fischer 384"),
                ]
            }
        ]
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
