/**
 * generate a random in range [_min_v_; _max_v_]
 * @param {number|null} min_v 
 * @param {number|null} max_v 
 * @returns 
 */
function rand(min_v,max_v){
    if(min_v==null){
        min_v=0
    }
    if(max_v==null){
        max_v=1
    }
    return Math.random() * (max_v - min_v) + min_v
}
/**
 * generate _n_ random numbers in range [_min_v_; _max_v_]
 * @param {number} n 
 * @param {number} min_v 
 * @param {number} max_v 
 * @returns 
 */
function nrand(n,min_v,max_v){
    let out=[]
    for(let i=0;i<n;i++){
        out.push(Math.exp(rand(min_v,max_v)))
    }
    return out
}
const data = [{
    x: [0, 50, 100, 150, 200, 255],
    y: nrand(6,0,100),
    type: 'line',
    name: "Fluo 405 nm Ex"
},{
    x: [0, 50, 100, 150, 200, 255],
    y: nrand(6,0,100),
    type: 'line',
    name: "Fluo 730 nm Ex"
},{
    x: [0, 50, 100, 150, 200, 255],
    y: nrand(6,10,200),
    type: 'line',
    name: "Fluo 688 nm Ex"
}]

const layout={
    autosize:true,

    showlegend: true,
    legend: {
        //orientation: 'h',
    },

    yaxis: {
        title:"relative frequency",
        type: 'log',
        autorange: true,
        // disable ticks
        showticklabels: false,
    },
    xaxis:{
        title: "relative intensity",
        tickvals: [0,50,100,150,200,255],
        ticktext: ["0","50","100","150","200","255"]
    },

    margin: {
        t:20, // top margin for pan/zoom buttons
        l:0, // reduced y axis margin
        b:40, // bottom margin for x-axis title
    },
}

const config={
    responsive: true,
    modeBarButtonsToRemove:['sendDataToCloud'],
    showLink:false,
    displaylogo:false,
}

let child = document.getElementById('histogram-panel');
if(!child){throw new Error("child is null")}

new ResizeObserver(function(){
    // @ts-ignore
    Plotly.relayout('histogram-panel', {autosize: true});
}).observe(child)

/// @ts-ignore
Plotly.newPlot('histogram-panel', data, layout, config);
