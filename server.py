import json
from flask import Flask, send_from_directory, request

from seaconfig import *

app = Flask(__name__, static_folder='src')

@app.route('/')
def index():
    # send local file "index.html" as response
    return send_from_directory('.', 'index.html')

@app.route(rule='/img/<path:path>')
def send_img_from_local_path(path):
    return send_from_directory('img', path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

@app.route('/src/<path:path>')
def send_js(path):
    return send_from_directory('src', path)

@app.route('/p/<path:path>')
def send_p(path):
    return send_from_directory('p', path)

remote_image_path_map:dict[str,dict]={
    "/server/data/acquisition/a65914/D05_2_3_1_1_fluo405.tiff":{
        "dir":"img",
        "filename":"cells.webp"
    },
    "/server/data/acquisition/a65914/D05_3_5_1_1_fluo405.tiff":{
        "dir":"img",
        "filename":"cells2.webp"
    },
}

@app.route("/api/img_remote")
def send_img_from_remote_storage():
    """
    send an image from remote storage
    """

    # get query parameters
    image_path=request.args.get("image_path")
    if image_path is None:
        return json.dumps({"status":"error","message":"no image_path provided"})

    if image_path not in remote_image_path_map:
        return json.dumps({"status":"error","message":"image_path not found"})

    return send_from_directory(remote_image_path_map[image_path]["dir"], remote_image_path_map[image_path]["filename"])

somemap:dict[str,dict]={}

# start acquisition
@app.route("/api/acquisition/start", methods=["POST"])
def start_acquisition():
    """
    start an acquisition with the given configuration
    """

    # get post data with key "config_file"
    json_data=None
    try:
        json_data=request.get_json()
    except Exception as e:
        pass

    if json_data is None:
        return json.dumps({"status": "error", "message": "no json data"})
    
    config = AcquisitionConfig.from_json(json_data["config_file"])
    print("starting acquisition with config:",config)

    # TODO generate some unique acqisition id to identify this acquisition by
    # must be robust against server restarts, and async requests
    # must also cache previous run results, to allow fetching information from past acquisitions
    acquisition_id="a65914"

    somemap[acquisition_id]={
        "acquisition_id":acquisition_id,
        "acquisition_config":config,
        "num_images_acquired":0,
        "num_images_total":1000,
    }

    return json.dumps({"status": "success","acquisition_id":acquisition_id})

@app.route("/api/acquisition/status", methods=["GET","POST"])
def get_acquisition_status():
    """
    get status of an acquisition
    """

    acquisition_id=None
    try:
        acquisition_id=request.get_json()["acquisition_id"]
    except Exception as e:
        pass

    if acquisition_id is None:
        return json.dumps({"status":"error","message":"no acquisition_id provided"})
    
    # TODO get status of acquisition with id "acquisition_id"
    # just mock something for now

    if acquisition_id not in somemap:
        return json.dumps({"status":"error","message":"acquisition_id is invalid"})

    acq_res=somemap[acquisition_id]
    acq_res["num_images_acquired"]+=1
    if acq_res["num_images_acquired"]>=acq_res["num_images_total"]:
        acq_res["num_images_acquired"]=acq_res["num_images_total"]

    ret={
        # request success
        "status":"success",

        "acquisition_id":acquisition_id,
        "acquisition_status":"running",
        "acquisition_progress":{
            # measureable progress
            "current_num_images":acq_res["num_images_acquired"],
            "time_since_start_s":50,
            "start_time_iso":"2021-01-01T12:00:00",
            "current_storage_usage_GB":4.3,

            # estimated completion time information
            # estimation may be more complex than linear interpolation, hence done on server side
            "estimated_total_time_s":100,

            # last image that was acquired
            "last_image":{
                "well":"D05",
                "site":{
                    "x":2,
                    "y":3,
                    "z":1,
                },
                "timepoint":1,
                "channel":"fluo405",
                "full_path":"/server/data/acquisition/a65914/D05_2_3_1_1_fluo405.tiff"
            },
        },

        # some meta information about the acquisition, derived from configuration file
        "acquisition_meta_information":{
            "total_num_images":acq_res["num_images_total"],
            "max_storage_size_images_GB":10,
        },

        "acquisition_config":{
            # full acquisition config copy here
        },

        "message":f"Acquisition is running, {100*acq_res['num_images_acquired']/acq_res['num_images_total']}% complete"
    }
    if acq_res["num_images_acquired"]==acq_res["num_images_total"]:
        ret["acquisition_status"]="finished"
        ret["message"]="Acquisition is finished"

    return json.dumps(ret)

# get objectives
@app.route("/api/get_features/hardware_capabilities", methods=["GET","POST"])
def get_hardware_capabilities():
    """
    get a list of all the hardware capabilities of the system

    these are the high-level configuration options that the user can select from
    """

    ret={
        "main_camera_objectives":[
            {
                "name":"4x Olympus",
                "handle":"4xolympus",
                "magnification":4,
            },
            {
                "name":"10x Olympus",
                "handle":"10xolympus",
                "magnification":10,
            },
            {
                "name":"20x Olympus",
                "handle":"20xolympus",
                "magnification":20,
            },
        ],
        "main_camera_triggers":[
            {
                "name":"Software",
                "handle":"software"
            },
            {
                "name":"Hardware",
                "handle":"hardware"
            },
        ],
        "main_camera_pixel_formats":[
            {
                "name":"8 Bit",
                "handle":"mono8"
            },
            {
                "name":"12 Bit",
                "handle":"mono12"
            },
        ],
        "wellplate_types":[
            {
                "name":"Perkin Elmer 96",
                "handle":"pe96",
                "num_rows":8,
                "num_cols":12,
            },
            {
                "name":"Falcon 96",
                "handle":"fa96",
                "num_rows":8,
                "num_cols":12,
            },
            {
                "name":"Perkin Elmer 384",
                "handle":"pe384",
                "num_rows":16,
                "num_cols":24,
            },
            {
                "name":"Falcon 384",
                "handle":"fa384",
                "num_rows":16,
                "num_cols":24,
            },
        ],
        "main_camera_imaging_channels":[c.to_dict() for c in [
            AcquisitionChannelConfig(
                name="Fluo 405 nm Ex",
                handle="fluo405",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="Fluo 488 nm Ex",
                handle="fluo488",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="Fluo 561 nm Ex",
                handle="fluo561",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="Fluo 688 nm Ex",
                handle="fluo688",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="Fluo 730 nm Ex",
                handle="fluo730",
                illum_perc=100,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="BF LED Full",
                handle="bfledfull",
                illum_perc=20,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="BF LED Right Half",
                handle="bfledright",
                illum_perc=20,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
            AcquisitionChannelConfig(
                name="BF LED Left Half",
                handle="bfledleft",
                illum_perc=20,
                exposure_time_ms=5.0,
                analog_gain=0,
                z_offset_um=0
            ),
        ]]
    }
    return json.dumps(ret)

@app.route("/api/get_features/machine_defaults", methods=["GET","POST"])
def get_machine_defaults():
    """
    get a list of all the low level machine settings

    these settings may be changed on the client side, for individual acquisitions
    (though clearly, this is highly advanced stuff, and may cause irreperable hardware damage!)
    """

    ret=[
        {
            "name":"laser autofocus exposure time [ms]",
            "value_kind":"number",
            "value":5.0
        },
        {
            "name":"laser autofocus camera analog gain",
            "value_kind":"number",
            "value":0
        },
        {
            "name":"laser autofocus camera pixel format",
            "value_kind":"option",
            "value":"mono8",
            "options":[
                {
                    "name":"8 Bit",
                    "handle":"mono8"
                },
                {
                    "name":"10 Bit",
                    "handle":"mono10"
                }
            ]
        },

        {
            "name":"main camera image width [px]",
            "value_kind":"number",
            "value":2500
        },
        {
            "name":"main camera image height [px]",
            "value_kind":"number",
            "value":2500
        },
    ]

    return json.dumps(ret)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
