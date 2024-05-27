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
        out.push(rand(min_v,max_v))
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
}];

const layout={
    title: 'Create a Static Chart',
    showlegend: true,
    legend: {
        orientation: 'h',
        y:-0.4,
    },
    margin: { t: 0,l:20,b:20,r:0},
    updatemenus: [{
        type: 'buttons',
        direction: 'left',
        x: 0.1,
        xanchor: 'center',
        y: -0.15,
        yanchor: 'top',
        buttons: [
            {
                method: 'relayout',
                args: ['yaxis.type', 'linear'],
                label: 'Y\' = Y'
            }, {
                method: 'relayout',
                args: ['yaxis.type', 'log'],
                label: 'Y\' = log(Y)'
            }
        ]
    }]
}

const config={
    responsive: true,
    // staticPlot: true,
    // displayModeBar: false,
};
let observer = new MutationObserver(function(mutations) {
    window.dispatchEvent(new Event('resize'));
});

let child = document.getElementById('histogram-panel');
if(!child){throw new Error("child is null")}
observer.observe(child, {attributes: true})

/// @ts-ignore
Plotly.newPlot('histogram-panel', data, layout, config);
