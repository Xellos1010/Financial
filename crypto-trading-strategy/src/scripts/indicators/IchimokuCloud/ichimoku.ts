// src/scripts/indicators/IchimokuCloud/ichimoku.ts
import { IndicatorBase } from '../indicatorBase';
import { IchimokuCloudInput, IchimokuCloudOutput } from '../types/ichimoku';

export class IchimokuCloud extends IndicatorBase {
    private high: number[];
    private low: number[];
    private close: number[];
    private conversionPeriods: number;
    private basePeriods: number;
    private spanBPeriods: number;
    private displacement: number;

    constructor(prices: IchimokuCloudInput, conversionPeriods: number, basePeriods: number, spanBPeriods: number, displacement: number) {
        super(prices.close);
        this.high = prices.high;
        this.low = prices.low;
        this.close = prices.close;
        this.conversionPeriods = conversionPeriods;
        this.basePeriods = basePeriods;
        this.spanBPeriods = spanBPeriods;
        this.displacement = displacement;
    }

    calculate(): IchimokuCloudOutput {
        const calculateDonchian = (len: number, high: number[], low: number[]) => {
            return high.map((h, index) => {
                if (index < len - 1) return NaN;
                const highSlice = high.slice(index - len + 1, index + 1);
                const lowSlice = low.slice(index - len + 1, index + 1);
                return (Math.max(...highSlice) + Math.min(...lowSlice)) / 2;
            });
        };

        const conversionLine = calculateDonchian(this.conversionPeriods, this.high, this.low);
        const baseLine = calculateDonchian(this.basePeriods, this.high, this.low);
        const leadLine1 = conversionLine.map((value, index) => (value + baseLine[index]) / 2);
        const leadLine2 = calculateDonchian(this.spanBPeriods, this.high, this.low);
        const laggingSpan = this.close.slice(0, this.close.length - this.displacement);

        return { conversionLine, baseLine, leadLine1, leadLine2, laggingSpan };
    }
}
