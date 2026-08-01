import { JsonLineFramer } from "../src/main/sidecar/JsonLineFramer.js";

describe("JsonLineFramer", () => {
  it("combines partial chunks and extracts multiple frames", () => {
    const framer = new JsonLineFramer();
    expect(framer.push(Buffer.from('{"one":'))).toEqual([]);
    expect(framer.push(Buffer.from('1}\n{"two":2}\n'))).toEqual([{ one: 1 }, { two: 2 }]);
    expect(() => framer.finish()).not.toThrow();
  });

  it("rejects oversized and incomplete frames", () => {
    const oversized = new JsonLineFramer(8);
    expect(() => oversized.push(Buffer.from("123456789"))).toThrow(/exceeds/);
    const partial = new JsonLineFramer();
    partial.push(Buffer.from("{}"));
    expect(() => partial.finish()).toThrow(/incomplete/);
  });
});
