import { initialOperationState, operationReducer } from "../src/renderer/app/OperationContext.js";
import type { WorkflowEvent } from "../src/shared/contracts.js";

function event(name: string, payload: Record<string, unknown>, sequence = 1): WorkflowEvent {
  return {
    type: "event",
    protocol: 1,
    event: name,
    queue_id: "queue-1",
    operation_id: "operation-1",
    job_id: "job-1",
    sequence,
    payload,
  };
}

describe("operationReducer", () => {
  it("tracks stages, text progress, and a bounded log", () => {
    let state = operationReducer(initialOperationState, event("workflow.started", {}));
    state = operationReducer(
      state,
      event("workflow.stage_started", { stage: "transcribe", item: "/media/first.wav" }, 2),
    );
    state = operationReducer(
      state,
      event(
        "workflow.stage_progress",
        { stage: "transcribe", percent: 42, message: "Frame 42", item: "/media/first.wav" },
        3,
      ),
    );
    state = operationReducer(state, event("workflow.stage_completed", { stage: "transcribe" }, 4));
    for (let index = 0; index < 110; index += 1) {
      state = operationReducer(state, event("workflow.log", { message: `line-${index}` }, index + 5));
    }
    expect(state.percent).toBe(42);
    expect(state.statusText).toBe("Frame 42");
    expect(state.stages.transcribe).toBe("completed");
    expect(state.currentItem).toBe("/media/first.wav");
    expect(state.items["/media/first.wav"]).toEqual({
      stage: "transcribe",
      percent: 100,
      statusText: "Transcribe",
    });
    expect(state.logs).toHaveLength(100);
    expect(state.logs[0]).toBe("line-10");
  });
});
