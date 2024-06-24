export const calculateIchimoku = (prices: { high: number[], low: number[], close: number[] }, conversionPeriods: number, basePeriods: number, spanBPeriods: number, displacement: number) => {
    const calculateDonchian = (len: number, high: number[], low: number[]) => {
        return high.map((h, index) => {
            if (index < len - 1) return NaN;
            const highSlice = high.slice(index - len + 1, index + 1);
            const lowSlice = low.slice(index - len + 1, index + 1);
            return (Math.max(...highSlice) + Math.min(...lowSlice)) / 2;
        });
    };

    const conversionLine = calculateDonchian(conversionPeriods, prices.high, prices.low);
    const baseLine = calculateDonchian(basePeriods, prices.high, prices.low);
    const leadLine1 = conversionLine.map((value, index) => (value + baseLine[index]) / 2);
    const leadLine2 = calculateDonchian(spanBPeriods, prices.high, prices.low);
    const laggingSpan = prices.close.slice(0, prices.close.length - displacement);
    
    return { conversionLine, baseLine, leadLine1, leadLine2, laggingSpan };
};
