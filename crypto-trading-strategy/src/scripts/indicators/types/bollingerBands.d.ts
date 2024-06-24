// src/scripts/indicators/types/bollingerBands.d.ts
export interface RegularBollingerBands {
    upperBand: number[];
    lowerBand: number[];
    basis: number[];
}

export interface FibonacciBollingerBands {
    upperBands: {
        upper_1: number[];
        upper_2: number[];
        upper_3: number[];
        upper_4: number[];
        upper_5: number[];
        upper_6: number[];
    };
    lowerBands: {
        lower_1: number[];
        lower_2: number[];
        lower_3: number[];
        lower_4: number[];
        lower_5: number[];
        lower_6: number[];
    };
    basis: number[];
}
