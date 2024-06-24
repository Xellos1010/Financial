export const calculateRSI = (prices: number[], period: number): number[] => {
    let rsi: number[] = [];
    let gains = 0;
    let losses = 0;
    for (let i = 1; i <= period; i++) {
        let change = prices[i] - prices[i - 1];
        if (change > 0) gains += change;
        else losses -= change;
    }
    let averageGain = gains / period;
    let averageLoss = losses / period;
    let rs = averageGain / averageLoss;
    rsi.push(100 - 100 / (1 + rs));

    for (let i = period + 1; i < prices.length; i++) {
        let change = prices[i] - prices[i - 1];
        if (change > 0) {
            averageGain = (averageGain * (period - 1) + change) / period;
            averageLoss = (averageLoss * (period - 1)) / period;
        } else {
            averageGain = (averageGain * (period - 1)) / period;
            averageLoss = (averageLoss * (period - 1) - change) / period;
        }
        rs = averageGain / averageLoss;
        rsi.push(100 - 100 / (1 + rs));
    }
    return rsi;
};
