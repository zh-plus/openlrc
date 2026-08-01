import { readFileSync } from "node:fs";
import path from "node:path";

const fixture = JSON.parse(
  readFileSync(path.join(process.cwd(), "tests", "fixtures", "protocol-v1.json"), "utf8"),
) as {
  queue: Record<string, unknown> & { entries: Array<Record<string, unknown>> };
  preflight: Record<string, unknown>;
  workflow_event: Record<string, unknown>;
};

import { preflightReportSchema, queueSnapshotSchema, workflowEventSchema } from "../src/shared/contracts.js";

describe("protocol v1 fixtures", () => {
  it("keeps queue, preflight, and workflow events aligned with Zod", () => {
    expect(queueSnapshotSchema.parse(fixture.queue).entries[0]?.queue_id).toBe("queue-1");
    expect(preflightReportSchema.parse(fixture.preflight).status).toBe("ready");
    expect(workflowEventSchema.parse(fixture.workflow_event).payload.percent).toBe(38);
  });

  it("rejects a queue state that is not part of protocol v1", () => {
    expect(() =>
      queueSnapshotSchema.parse({
        ...fixture.queue,
        entries: [{ ...fixture.queue.entries[0], state: "waiting" }],
      }),
    ).toThrow();
  });
});
