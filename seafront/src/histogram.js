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

/**@type{PlotlyLayout} */
const layout={
    autosize:true,

    showlegend: true,

    yaxis: {
        title:"relative frequency ( Log )",
        type: 'log',
        autorange: true,
        fixedrange:true,
        // disable ticks
        showticklabels: false,
    },
    xaxis:{
        title: "relative intensity",
        autorange:true,
        fixedrange:true,
        tickvals: [0,50,100,150,200,255],
        ticktext: ["0","50","100","150","200","255"]
    },

    margin: {
        t:20, // top margin for pan/zoom buttons
        l:30, // reduced y axis margin
        b:40, // bottom margin for x-axis title
    },
    plot_bgcolor:"lightgrey",
    paper_bgcolor:"lightgrey"
}

/**@type{PlotlyConfig} */
const config={
    responsive: true,
    modeBarButtonsToRemove:[
        'sendDataToCloud',
        "zoom2d","pan2d","select2d","lasso2d",
        "zoomIn2d","zoomOut2d",
        "autoScale2d","resetScale2d"
    ],
    showLink:false,
    displaylogo:false,
}

const histogram_plot_element_id='histogram-panel'
/** @ts-ignore @type{PlotlyHTMLElement|null} */
const histogram_plot_element = document.getElementById(histogram_plot_element_id);
if(!histogram_plot_element){throw new Error("child is null")}

new ResizeObserver(function(){
    // @ts-ignore
    Plotly.relayout(histogram_plot_element_id, {autosize: true});
}).observe(histogram_plot_element)

/**
 * @typedef{{title?:string,type?:("log"),autorange?:boolean,fixedrange?:boolean,showticklabels?:boolean,tickvals?:number[],ticktext?:string[]}} PlotlyAxis
 * @typedef{{autosize?:boolean,showlegend?:boolean,yaxis?:PlotlyAxis,xaxis?:PlotlyAxis,margin?:{t?:number,b?:number,l?:number,r?:number},plot_bgcolor?:string,paper_bgcolor?:string}} PlotlyLayout
 * @typedef{{responsive?:boolean,modeBarButtonsToRemove?:("sendDataToCloud"|"zoom2d"|"pan2d"|"select2d"|"lasso2d"|"zoomIn2d"|"zoomOut2d"|"autoScale2d"|"resetScale2d")[],showLink?:boolean,displaylogo?:boolean}} PlotlyConfig
 * @typedef{(HTMLElement&{data:{name:string}[]})}PlotlyHTMLElement
 * @typedef{{color?:string}}PlotElementStyle
 * @typedef{{x?:number[],y?:number[],type?:("scatter"),name?:string,legendrank?:number,mode?:"lines"|"markers"|"lines+markers",line?:PlotElementStyle,marker?:PlotElementStyle}}PlotlyTrace
 * 
 * ref https://plotly.com/javascript/plotlyjs-function-reference/
 * 
 * @ts-ignore @type{{
 * newPlot:function(string,any[]?,PlotlyLayout?,PlotlyConfig?):void,
 * deleteTraces:function(PlotlyHTMLElement,number[]):void,
 * addTraces:function(PlotlyHTMLElement,PlotlyTrace):void,
 * }}
 */
let Plotly=window.Plotly

Plotly.newPlot(histogram_plot_element_id, [], layout, config)
