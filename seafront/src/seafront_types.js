
/**
 * @typedef {{
    x:number,
    y:number,
    z:number,
 * }} WellSite

* @typedef {{
    current_num_images:number,
    time_since_start_s:number,
    start_time_iso:string,
    current_storage_usage_GB:number,

    estimated_total_time_s:number|null,

    last_image:ImageStoreInfo
 * }} AcquisitionProgressStatus

 * @typedef {{
    total_num_images:number,
    max_storage_size_images_GB:number,
 * }} AcquisitionMetaInformation

 * @typedef {{
    row:number,
    col:number,
    selected:boolean,
 * }} AcquisitionWellSiteConfigurationSiteSelectionItem

 * @typedef {{
    h:number,
    m:number,
    s:number,
 * }} AcquisitionWellSiteConfigurationDeltaTime

 * @typedef {{
    num_x:number,
    delta_x_mm:number,
    num_y:number,
    delta_y_mm:number,
    num_t:number,
    delta_t:AcquisitionWellSiteConfigurationDeltaTime,

    mask:AcquisitionWellSiteConfigurationSiteSelectionItem[],
 * }} AcquisitionWellSiteConfiguration

 * @typedef {{
    row:number,
    col:number,
    selected:boolean,
 * }} PlateWellConfig

 * @typedef {{
    name:string,
    handle:string,
    info:*
    }} ConfigItemOption

 * @typedef {{
    name:string,
    handle:string,
    value_kind:"number"|"text"|"option"|"action",
    value:number|string,
    frozen:boolean,
    options:(ConfigItemOption[])|null
 * }} ConfigItem

 * @typedef {{
    major:number,
    minor:number,
    patch:number,
 * }} Version

 * @typedef {{
    project_name:string,
    plate_name:string,
    cell_line:string,
    
    grid:AcquisitionWellSiteConfiguration,

    wellplate_type:string,
    plate_wells:PlateWellConfig[],

    channels:AcquisitionChannelConfig[],

    autofocus_enabled:boolean,

    machine_config:(ConfigItem[])?,

    comment:string|null,

    spec_version:Version,

    timestamp:string|null,
 * }} AcquisitionConfig

 * @typedef {{
    acquisition_id:string,
    acquisition_status:"running"|"cancelled"|"completed"|"crashed",
    acquisition_progress:AcquisitionProgressStatus,

    acquisition_meta_information:AcquisitionMetaInformation,

    acquisition_config:AcquisitionConfig,

    message:string,
 * }} AcquisitionStatusOut
 
 * @typedef {{}} AcquisitionCancelResponse

 * @typedef {{
    Manufacturer:string,
    Model_name:string,
    Model_id_manufacturer:string,

    Model_id:string,

    Offset_A1_x_mm:number,
    Offset_A1_y_mm:number,

    Offset_bottom_mm:number,

    Well_distance_x_mm:number,
    Well_distance_y_mm:number,

    Well_size_x_mm:number,
    Well_size_y_mm:number,

    Num_wells_x:number,
    Num_wells_y:number,

    Length_mm:number,
    Width_mm:number,

    Well_shape:*
 * }} Wellplate

 * @typedef {{
    wellplate_types:Wellplate[],
    main_camera_imaging_channels:AcquisitionChannelConfig[],
 * }} HardwareCapabilitiesResponse

 * @typedef {{
    name:string,
    handle:string,

    illum_perc:number,
    exposure_time_ms:number,
    analog_gain:number,

    z_offset_um:number,

    num_z_planes:number,
    delta_z_um:number,
    
    enabled:boolean,
 * }} AcquisitionChannelConfig

 * @typedef {{
    x_pos_mm:number,
    y_pos_mm:number,
    z_pos_mm:number,
 * }} Position

 * @typedef {{
    channel:AcquisitionChannelConfig,
    width_px:number,
    height_px:number,
    timestamp:number,
    position:SitePosition,
    storage_path:string|null,
 * }} ImageStoreInfo

    @typedef {{
        well_name:string,
        site_x:number,site_y:number,site_z:number,
        x_offset_mm:number,y_offset_mm:number,z_offset_mm:number,
        position:Position,
    }} SitePosition

@typedef {"idle"|"channel_snap"|"channel_stream"|"loading_position"|"moving"} CoreState

 * @typedef {{
    state:CoreState,
    is_in_loading_position:boolean,
    stage_position:Position,
 * }} AdapterState
 * 
 * @typedef {{
    adapter_state:AdapterState,

    // python dictionary
    latest_imgs:Record<string,ImageStoreInfo|null>,
    current_acquisition_id:string|null,
 * }} CoreCurrentState
 */