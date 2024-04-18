
p.plate_wells=make_observable([])
/**
 * 
 * @param {Info} info 
 */
p.initwellnavigator=function(info){
    let element=info.element || info

    let plate_type=p.config.wellplate_type.valueOf()
    
    let num_cols=0
    let num_rows=0

    // if plate type ends in 96, then 8x12
    if(plate_type.endsWith("96")){
        num_cols=12
        num_rows=8
    }else if(plate_type.endsWith("384")){
        num_cols=24
        num_rows=16
    }

    // add 1 for headers
    num_cols+=1
    num_rows+=1

    element.style.setProperty("--num-cols",num_cols);
    element.style.setProperty("--num-rows",num_rows);

    let new_plate_wells=[]
    
    for(let i=0;i<num_rows;i++){
        for(let j=0;j<num_cols;j++){
            let new_well = {row:i,col:j}
            
            new_plate_wells.push(new_well);
        }
    }

    p.plate_wells.withPaused((w)=>{
        w.splice(0,w.length,...new_plate_wells)
    })
}