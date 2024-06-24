// src/scripts/indicators/types/ichimoku.d.ts
export interface IchimokuCloudInput {
    high: number[];
    low: number[];
    close: number[];
}

export interface IchimokuCloudOutput {
    conversionLine: number[];
    baseLine: number[];
    leadLine1: number[];
    leadLine2: number[];
    laggingSpan: number[];
}
