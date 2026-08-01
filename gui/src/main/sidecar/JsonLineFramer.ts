export class JsonLineFramer {
  private buffer = Buffer.alloc(0);

  constructor(private readonly maxFrameBytes = 1024 * 1024) {}

  push(chunk: Buffer): unknown[] {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    if (this.buffer.length > this.maxFrameBytes && !this.buffer.includes(0x0a)) {
      throw new Error("Sidecar frame exceeds the 1 MiB limit.");
    }
    const frames: unknown[] = [];
    while (true) {
      const newline = this.buffer.indexOf(0x0a);
      if (newline < 0) break;
      if (newline > this.maxFrameBytes) throw new Error("Sidecar frame exceeds the 1 MiB limit.");
      const line = this.buffer.subarray(0, newline);
      this.buffer = this.buffer.subarray(newline + 1);
      if (line.length === 0) continue;
      frames.push(JSON.parse(line.toString("utf8")) as unknown);
    }
    return frames;
  }

  finish(): void {
    if (this.buffer.length > 0) throw new Error("Sidecar ended with an incomplete JSON frame.");
  }
}
