"use strict";

/** @typedef {import('./init.js').AlpineMicroscopeState} AlpineMicroscopeState */

/**@type {PlotlyLayout} */
export const histogramLayout = {
    autosize: true,

    showlegend: true,
    // https://stackoverflow.com/questions/39668369/listing-legend-items-horizontally-and-centered-below-a-plot#42729056
    // which links to https://plotly.com/r/reference/#layout-legend
    legend: {
        orientation: "h",
        xanchor: "center",
        x: 0.5,
        y: -0.2,
    },

    yaxis: {
        title: {text:"relative frequency ( Log )",standoff:0,},
        type: 'log',
        autorange: true,
        autorangeoptions:{
            minallowed:Math.log(1),
        },
        // disable ticks
        showticklabels: false,
    },
    xaxis: {
        title: {text:"relative intensity",},
        autorange: true,
        fixedrange: true,
        range:[0,256],
        tickvals: [0, 50, 100, 150, 200, 255],
        ticktext: ["0", "50", "100", "150", "200", "255"]
    },

    margin: {
        t: 20, // top margin for pan/zoom buttons
        l: 20, // reduced y axis margin
        r: 20, // reduced x axis margin
        b: 90, // bottom margin for x-axis title
    },
    plot_bgcolor: "lightgrey",
    paper_bgcolor: "lightgrey"
}

/**@type {PlotlyConfig} */
export const histogramConfig = {
    responsive: true,
    modeBarButtonsToRemove: [
        'sendDataToCloud',
        "zoom2d", "pan2d", "select2d", "lasso2d",
        "zoomIn2d", "zoomOut2d",
        "autoScale2d", "resetScale2d"
    ],
    showLink: false,
    displaylogo: false,
}

/**
 * 
 * @param {HTMLElement} el 
 */
export function makeHistogram(el){
    Plotly.newPlot(el,[
        {
            type:"scatter",
            x:[0, 1,  2,  3,  4, 255],
            y:[0, 10, 4, 13, -2, 5],
            mode:"lines+markers"
        }
    ],histogramLayout,histogramConfig);

    new ResizeObserver(function () {
        // @ts-ignore
        Plotly.relayout(el, { autosize: true });
    }).observe(el)
}

/**
 * Update histogram display with cached channel image data.
 * Builds histogram traces for each enabled channel and renders with Plotly.
 * @this {AlpineMicroscopeState}
 * @param {HTMLElement} el - The element to render the histogram into
 */
export function updateHistogram(el) {
    /** @type {PlotlyTrace[]} */
    const data = [];
    const xvalues = new Uint16Array(257).map((v, i) => i);

    for (const [key, value] of Array.from(
        this.cached_channel_image.entries(),
    ).toSorted((l, r) => {
        return l[0] > r[0] ? 1 : -1;
    })) {
        // key is handle, i.e. key===value.info.channel.handle (which is not terribly useful for displaying)
        const name = value.info.channel.name;

        // skip channels that are not enabled
        if (
            !(
                this.microscope_config.channels.find(
                    (c) => c.handle == key,
                )?.enabled ?? false
            )
        ) {
            continue;
        }

        const y = new Float32Array(xvalues.length);

        const rawdata = (() => {
            switch (value.bit_depth) {
                case 8:
                    return new Uint8Array(value.data);
                case 16:
                    return new Uint16Array(value.data).map(
                        (v) => v >> 8,
                    );
                default:
                    throw new Error(``);
            }
        })();
        for (const val of rawdata) {
            y[val]++;
        }
        data.push({
            name,
            // @ts-ignore
            x: xvalues,
            // @ts-ignore
            y,
        });
    }
    Plotly.react(el, data, histogramLayout, histogramConfig);
}
