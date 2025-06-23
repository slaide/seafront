//declare module "plotly"{
declare global{
    /**
     * can use html syntax.
     */
    type PlotlyText=string;
    /**
     * connect data points by:
     * - spline: spline interpolation
     * - vhv: vertical line to midway Y, then horizontal line to final X, then vertical line to final Y
     * - hvh: analog to vhv
     * - vh: veritical line to final Y, then horizontal line to final X
     * - hv: analog to vh 
     * - linear: linear interpolation
     */
    type PlotlyShape="spline"|"vhv"|"hvh"|"vh"|"hv"|"linear";
    /**
     * css color name, i.e.:
     * 
     * - value can be name, e.g. 'red'
     * - or hex code, e.g. '#17BECF'
     * - or rgb(219, 64, 82)
     * */
    type PlotlyColor=string;
    type PlotlySizeMode="area";
    type PlotlySymbol='circle'|'square'|'diamond'|'cross';
    type PlotElementStyle={
        /** can be either same value for all items, or one value per item */
        color?:string|string[];
        /** can be either same value for all items, or one value per item */
        size?:number|number[];
        width?:number|number[];
        shape?:PlotlyShape|PlotlyShape[];
        /** border thickness */
        line?:number|number[]|PlotElementStyle|PlotElementStyle[];
        dash?: "solid"|"dash"|"dot"|"dashdot";
        sizeref?: number;
        sizemode?: PlotlySizeMode|PlotlySizeMode[];
        symbol?:PlotlySymbol|PlotlySymbol[];
    };

    type PlotlyTextPosition='top center'|'auto';
    type PlotlyFont='Times New Roman'|'Arial, sans-serif';
    type PlotlyTrace=(
        (
            {
                x?:number[]|string[];
                y?:number[]|string[];
                mode?:"lines"|"markers"|"lines+markers";
                orientation?: "h"|"horizontal";
                /** text visible on hover */
                text?: PlotlyText|PlotlyText[];
            }&({
                /** https://plotly.com/javascript/line-charts/ */
                type?:"scatter";
            }|{
                /** https://plotly.com/javascript/histograms/ */
                type?:"histogram";
                histfunc?: "count"|"sum";
                autobinx?: boolean;
                histnorm?: "count"|"probability";
                xbins?: {
                    end: number;
                    size: number;
                    start: number;
                };
                cumulative?: {enabled: boolean;};
            }|{
                /** https://plotly.com/javascript/bar-charts/ */
                type?:"bar";
                base?: number|number[];
                /** e.g. '%{base}' */
                hovertemplate?: string;
            })
        )|{
            /** https://plotly.com/javascript/pie-charts/ */
            type?: 'pie';
            values?: number[];
            labels?: string[];
            domain?: {
                x: number[];
                y: number[];
            };
            hole?: number;
        }
    )&{
        opacity?: number;
        textposition?: PlotlyTextPosition|PlotlyTextPosition[];
        textfont?: PlotlyFontConfig;
        hoverinfo?: 'none'|'label+percent+name',
        textinfo?: 'none'|"label+percent";
        /** name used for this trace in the legend */
        name?:string;
        legendrank?:number;
        line?:PlotElementStyle;
        marker?:PlotElementStyle;
        insidetextorientation?: "radial";
        /** e.g. x2 ( for subplot at x=2, y=y) */
        xaxis?: string;
        /** e.g. y2 ( for subplot at x=x, y=2) */
        yaxis?: string;
    };

    type PlotlyFontConfig={
        family?: PlotlyFont;
        size?: number;
        color?: PlotlyColor;
    };
    type AnchorPosition='center'|'left'|'bottom';
    type PlotlyLayout={
        barmode?: 'group'|"stack";
        scattermode?: 'group';
        barcornerradius?: number;
        bargap?: number;
        bargroupgap?: number;
        title?: Label;
        width?:number;
        height?:number;
        autosize?:boolean;
        showlegend?:boolean;
        legend?: {
            y?: number;
            x?: number;
            yref?: 'paper';
            font?: PlotlyFontConfig;
            traceorder?: 'reversed';
            bgcolor?:PlotlyColor;
            bordercolor?:PlotlyColor;
            orientation?:"h"|"v";
            xanchor?:AnchorPosition;
            yanchor?:AnchorPosition;
        };

        yaxis?:PlotlyAxis;
        xaxis?:PlotlyAxis;

        margin?:{
            t?:number;
            b?:number;
            l?:number;
            r?:number;
            autoexpand?:boolean;
        };
        plot_bgcolor?:string;
        paper_bgcolor?:string;
        font?:PlotlyFontConfig;
        annotations?:{
            xref?: 'paper';
            yref?: 'paper';
            x?: number;
            y?: number;
            xanchor?: AnchorPosition;
            yanchor?: AnchorPosition;
            text?: PlotlyText;
            font?: PlotlyFontConfig;
            showarrow?: boolean;
        }[];
        /** https://plotly.com/javascript/subplots/ */
        grid?: {
            rows: number;
            columns: number;
            pattern: 'independent';
        };
    };
    type PlotlyModeBarButton=
        // for 2d charts
        "zoom2d"|"pan2d"|"select2d"|"lasso2d"|"zoomIn2d"|"zoomOut2d"|"autoScale2d"|"resetScale2d"|
        // for 3d charts
        "zoom3d"|"pan3d"|"orbitRotation"|"tableRotation"|"handleDrag3d"|"resetCameraDefault3d"|"resetCameraLastSave3d"|"hoverClosest3d"|
        // for cartesian charts
        "hoverClosestCartesian"|"hoverCompareCartesian"|
        // for geo charts
        "zoomInGeo"|"zoomOutGeo"|"resetGeo"|"hoverClosestGeo"|
        // 'other'
        "hoverClosestGl2d"|"hoverClosestPie"|"toggleHover"|"resetViews"|"toImage"|"sendDataToCloud"|"toggleSpikelines"|"resetViewMapbox"|
        {
            name: string;
            icon: PlotlyIcon,
            click:(plot:PlotlyPlot)=>void;
            direction?: 'up';
        };
    type PlotlySVGPath=string;
    type PlotlyIcon={
        width:number;
        height:number;
        path:PlotlySVGPath;
    };
    type PlotlyIconPresetPalette={
        get pencil():PlotlyIcon;
    };
    const Icons:PlotlyIconPresetPalette;
    
    type PlotlyConfig={
        showEditInChartStudio?: boolean;
        responsive?:boolean;
        modeBarButtonsToRemove?:PlotlyModeBarButton[];
        modeBarButtonsToAdd?: PlotlyModeBarButton[];
        showLink?:boolean;
        displaylogo?:boolean;
        displayModeBar?:boolean;
        toImageButtonOptions?: {
            format: "png"|"svg"|"jpeg"|"webp";
            filename: string;
            height: number;
            width: number;
            /** Multiply title/legend/axis/canvas sizes by this factor */
            scale: number;
        };
        /** used to upload the plot to. e.g. "https://chart-studio.plotly.com" */
        plotlyServerURL?: string;
        linkText?: string;
        doubleClickDelay?: number;
    };
    type PlotlyPlot=(HTMLElement&{
        data:{name:string}[];
        layout:PlotlyLayout;
    });

    type Label={
        text?:PlotlyText;
        standoff?:number;
    }
    type PlotlyAxis={
        title?:Label;
        type?:"log"|"date";
        fixedrange?:boolean;
        /** min value allowed in range. Even explicitly setting range:[a,b] can't go past this. */
        minallowed?:number;
        /** max value allowed in range. Even explicitly setting range:[a,b] can't go past this.
    must be more than minallowed */
        maxallowed?:number;
        /**
         * useful in combination with `fixedrange`.
         * 
         * indicate axis lower and higher limit (i.e. should have length 2)
         * 
         * allowed to leave either or both elements `null`, this impacts the default `autorange`.
         * */
        range?:number[];
        /** if 'min' or 'max' is in the value, only autorange that end, use existing `range` entry for the other end
    if range[0] is missing, 'min' is the default and false, 'max', and 'min reversed' are prohibited
    if range[1] is missing, 'max' is the default and false, 'min', and 'max reversed' are prohibited */
        autorange?:boolean|'min'|'max'|'min reversed'|'max reversed'|'reversed';
        /**
         * useful in combination with `autorange`.
         * 
         * allows setting range limits for autorange.
         * (requires manual scaling if axis type is not linear, e.g. if type=log -> autorangeoptions.minallow=Math.log(actualLowerLimit))
         * 
         * https://github.com/plotly/plotly.js/pull/6547#issuecomment-1546236028
         * */
        autorangeoptions?:{
            /** do not let min autorange go below this. Ignore if autorange = false, 'max', 'max reversed',
        or if it's <axis.minallowed or >axis.maxallowed */
            minallowed?:number;
            /** do not let max autorange go above this. Ignore if autorange = false, 'min', 'min reversed',
        or if it's <axis.minallowed or >axis.maxallowed */
            maxallowed?:number;
            /** ensure the autorange min <= and max >= these values,
        ignore any num outside either set of min/maxallowed constraints */
            include?: number|number[];
        };

        showline?:boolean;
        showgrid?:boolean;
        gridwidth?:number;
        showticklabels?:boolean;
        linecolor?:PlotlyColor;
        linewidth?:number;
        tickmode?:"linear";
        ticks?:"outside";
        tickcolor?:PlotlyColor;
        tickwidth?:number;
        ticklen?:number;
        tickfont?:PlotlyFontConfig;
        tickvals?:number[];
        ticktext?:string[];
        tickangle?:number;
        zeroline?: boolean;

        domain?:number[];
        anchor?:string;
    };

    type PlotlyValidationError={
        /** error message */      
        msg:string;
    };

    /**
     * https://plotly.com/javascript/
     * 
     * https://plotly.com/javascript/plotlyjs-function-reference/
     * */
    export class Plotly{
        /**
         * Draws a new plot in an <div> element, overwriting any existing plot. To update an existing plot in a <div>, it is much more efficient to use Plotly.react than to overwrite it.
         * @param element_id 
         * @param [data=[]] 
         * @param [layout={}] 
         * @param [config={}] 
         */
        static newPlot(
            element_id:string|HTMLElement,
            data?:PlotlyTrace[],
            layout?:PlotlyLayout,
            config?:PlotlyConfig,
        ):void;
        /**
         * Plotly.react has the same signature as Plotly.newPlot above, and can be used in its place to create a plot, but when called again on the same <div> will update it far more efficiently than Plotly.newPlot, which would destroy and recreate the plot. Plotly.react is as fast as Plotly.restyle/Plotly.relayout documented below.

        Important Note: In order to use this method to plot new items in arrays under data such as x or marker.color etc, these items must either have been added immutably (i.e. the identity of the parent array must have changed) or the value of layout.datarevision must have changed.
         * @param element_id 
         * @param [data=[]] 
         * @param [layout={}] 
         * @param [config={}] 
         */
        static react(
            element_id:string|HTMLElement,
            data?:PlotlyTrace[],
            layout?:PlotlyLayout,
            config?:PlotlyConfig,
        ):void;
        /**
         * This function has comparable performance to Plotly.react and is faster than redrawing the whole plot with Plotly.newPlot.

        This allows you to remove traces from an existing graphDiv by specifying the indices of the traces to be removed.
         * @param plot 
         * @param data list of indices of the traces to be removed (e.g. -1 for last trace, 0 for first trace)
         */
        static deleteTraces(
            plot:PlotlyPlot,
            indices:number|number[],
        ):void;
        /**
         * This function has comparable performance to Plotly.react and is faster than redrawing the whole plot with Plotly.newPlot.

        This allows you to add new traces to an existing graphDiv at any location in its data array. Every graphDiv object has a data component which is an array of JSON blobs that each describe one trace. The full list of trace types can be found in the Full Reference.
         * @param plot 
         * @param traces
         */
        static addTraces(
            plot:PlotlyPlot,
            traces:PlotlyTrace|PlotlyTrace[],
        ):void;
        /**
         * This function has comparable performance to Plotly.react and is faster than redrawing the whole plot with Plotly.newPlot.
        
        This allows you to add data to traces in an existing graphDiv.
         * @param plot
         * @param traces must have same length as `indices`
         * @param indices must have same length as `traces`
         * @param max_num_added_dates maximum number of dates added to each trace (truncates input, I guess?)
         */
        static extendTraces(
            plot:PlotlyPlot,
            traces:PlotlyTrace[],
            indices:number[],
            max_num_added_dates?:number,
        ):void;
        /**
         * Plotly.validate allows users to validate their input data array and layout object. This can be done on the data array and layout object passed into Plotly.newPlot or on an updated graphDiv with Plotly.validate(graphDiv.data, graphDiv.layout).
         */
        static validate(
            data:PlotlyTrace[],
            layout:PlotlyLayout,
        ):PlotlyValidationError[];
        /**
         * Using purge will clear the div, and remove any Plotly plots that have been placed in it.
         */
        static purge(
            plot:PlotlyPlot,
        ):void;
    }
}

// this line ensures that the 'declare global' are visible by the LSP in other .js files
export { }