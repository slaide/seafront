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

    channels:[
        {
            name:"Fluo 405 nm Ex",
            handle:"fluo405",
            illum_perc:100,
            exposure_time_ms:5.0,
            analog_gain:0,
            z_offset_um:0,
        },
        {
            name:"Fluo 488 nm Ex",
            handle:"fluo488",
            illum_perc:100,
            exposure_time_ms:5.0,
            analog_gain:0,
            z_offset_um:0,
        },
        {
            name:"Fluo 561 nm Ex",
            handle:"fluo561",
            illum_perc:100,
            exposure_time_ms:5.0,
            analog_gain:0,
            z_offset_um:0,
        },
        {
            name:"Fluo 688 nm Ex",
            handle:"fluo688",
            illum_perc:100,
            exposure_time_ms:5.0,
            analog_gain:0,
            z_offset_um:0,
        },
        {
            name:"Fluo 730 nm Ex",
            handle:"fluo730",
            illum_perc:100,
            exposure_time_ms:5.0,
            analog_gain:0,
            z_offset_um:0,
        },
        {
            name:"BF LED Full",
            handle:"bfledfull",
            illum_perc:20,
            exposure_time_ms:5.0,
            analog_gain:0,
            z_offset_um:0,
        },
        {
            name:"BF LED Right Half",
            handle:"bfledright",
            illum_perc:20,
            exposure_time_ms:5.0,
            analog_gain:0,
            z_offset_um:0,
        },
        {
            name:"BF LED Left Half",
            handle:"bfledleft",
            illum_perc:20,
            exposure_time_ms:5.0,
            analog_gain:0,
            z_offset_um:0,
        },
    ]
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

const main_camera_triggers=[
    {
        name:"Software",
        handle:"software"
    },
    {
        name:"Hardware",
        handle:"hardware"
    }
]
const main_camera_pixel_types=[
    {
        name:"8 bit",
        handle:"mono8"
    },
    {
        name:"12 bit",
        handle:"mono12"
    }
]
const objectives=[
    {
        name:"4x Olympus",
        handle:"4xolympus",
    },
    {
        name:"10x Olympus",
        handle:"10xolympus",
    },
    {
        name:"20x Olympus",
        handle:"20xolympus",
    },
]
/**
 * return name of an objective when given its handle
 * @param {string} handle 
 * @returns 
 */
const objective_handle2name=function(handle){
    for(let o of objectives){
        if(o.handle==handle){
            return o.name
        }
    }
    return null
}

const wellplate_types=[
    {
        name:"96 Well Plate",
        entries:[
            {
                name:"Perkin Elmer 96",
                handle:"pe96"
            },
            {
                name:"Falcon 96",
                handle:"fa96"
            }
        ]
    },
    {
        name:"384 Well Plate",
        entries:[
            {
                name:"Perkin Elmer 384",
                handle:"pe384"
            },
            {
                name:"Falcon 384",
                handle:"fa384"
            },
            {
                name:"Thermo Fischer 384",
                handle:"tf384"
            }
        ]
    }
]
/**
 * 
 * @param {HTMLElement} tab_header 
 * @returns 
 */
const init_tab_header=function(tab_header){
    let tab_header_children=tab_header.querySelectorAll("*[target]")
    
    /** @type HTMLElement[] */
    let valid_tab_children=[]
    tab_header_children.forEach((el)=>{
        if(!(el instanceof HTMLElement)){return}
        
        let element_target_id=el.getAttribute("target")
        if(!element_target_id){console.error("element target is null");return}
        let tab_target=document.getElementById(element_target_id)
        if(!tab_target){
            console.error("tab header target '"+el.getAttribute("target")+"' not found",el);
            return
        }
        tab_target.classList.add("hidden");

        valid_tab_children.push(el);
        el.addEventListener("click",tab_head_click);
    });
    if(valid_tab_children.length==0){
        return
    }
    valid_tab_children[0].click()
}
let _tabHeadMap_currentTarget=new Map()
/**
 * 
 * @param {MouseEvent} e 
 */
const tab_head_click=function(e){
    let head=e.currentTarget;
    if(!head){return}
    if(!(head instanceof HTMLElement)){return}
    if(!head.parentNode){return}

    let current_target=_tabHeadMap_currentTarget.get(head.parentNode)
    if(current_target){
        current_target.classList.add("hidden");
    }

    head.parentNode.querySelectorAll("*").forEach((el)=>{
        el.classList.remove("active")
    });

    head.classList.add("active")

    let target = head.getAttribute("target")
    if(!target){console.error("target is null");return}
    let target_el = document.getElementById(target)
    if(!target_el){console.error("target element not found");return}

    _tabHeadMap_currentTarget.set(head.parentNode,target_el);
    
    target_el.classList.remove("hidden");
}
