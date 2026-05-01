import { create } from "zustand";

type UiState = {
  replayAutoplay: boolean;
  axisView: string;
  axisScale: string;
  setReplayAutoplay: (value: boolean) => void;
  setAxisView: (value: string) => void;
  setAxisScale: (value: string) => void;
};

export const useUiStore = create<UiState>((set) => ({
  replayAutoplay: false,
  axisView: "own",
  axisScale: "raw",
  setReplayAutoplay: (replayAutoplay) => set({ replayAutoplay }),
  setAxisView: (axisView) => set({ axisView }),
  setAxisScale: (axisScale) => set({ axisScale })
}));
