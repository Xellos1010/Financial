export type MovingAverageType = 'SMA' | 'EMA' | 'DEMA' | 'TEMA' | 'VWMA' | 'ZLEMA' | 'WMA' | 'HMA' | 'RMA';

export interface MovingAverageInput {
    prices: number[];
    period: number;
    volumes?: number[]; // Only needed for VWMA
}

export interface MovingAverageOutput {
    result: number[];
}
