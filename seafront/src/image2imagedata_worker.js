
/**
 * 
 * @param {{width:number,height:number}} payload
 * @param {any} image_bytes
 * @returns {ImageData}
 */
function image_data_to_imagedata(payload,image_bytes){
    const rawbytes=new Uint8Array(image_bytes)

    let rgba=new Uint8ClampedArray()
    if((rawbytes.length/(payload.width*payload.height))==1){
        const mono=rawbytes

        rgba=new Uint8ClampedArray(4*payload.width*payload.height)
        for(let i=0;i<mono.length;i++){
            // clamp happens inside the array
            const pix=mono[i]

            rgba[i*4+0]=pix
            rgba[i*4+1]=pix
            rgba[i*4+2]=pix
            rgba[i*4+3]=255
        }
    }else{
        const mono=new Uint16Array(image_bytes)

        rgba=new Uint8ClampedArray(4*payload.width*payload.height)
        for(let i=0;i<mono.length;i++){
            // convert u16 to u8 by shifting
            // clamp happens inside the array
            const pix=(mono[i]>>8)

            rgba[i*4+0]=pix
            rgba[i*4+1]=pix
            rgba[i*4+2]=pix
            rgba[i*4+3]=255
        }
    }
    const imagedata=new ImageData(rgba,payload.width,payload.height)
    return imagedata
}

// receive data as message event
self.addEventListener('message', (event) => {
    self.postMessage(image_data_to_imagedata(event.data.payload,event.data.image_bytes))
    self.close()
})