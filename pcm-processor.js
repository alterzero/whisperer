class PcmProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (ch?.length > 0) this.port.postMessage(new Float32Array(ch));
    return true;
  }
}
registerProcessor("pcm-processor", PcmProcessor);
