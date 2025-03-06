
// takes ca. 20ms on 3ghz rpi5
/**
 * 
 * @param {ImageData} img 
 * @returns {number[]}
 */
function calculate_histogram(img){
    let ret=new Array(256)
    for(let i=0;i<256;i++)ret[i]=0
    // image data has 4 components
    for(let i=0;i<img.data.length;i+=4){
        ret[img.data[i]]++
    }
    return ret
}

// receive data as message event
self.addEventListener('message', (event) => {
    self.postMessage(calculate_histogram(event.data))
    self.close()
})
  