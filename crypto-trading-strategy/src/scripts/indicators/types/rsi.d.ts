// src/scripts/indicators/types/rsi.d.ts
export interface RSIInput {
    prices: number[];
    period: number;
}

export interface RSIOutput {
    rsi: number[];
}
