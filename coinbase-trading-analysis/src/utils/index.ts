// src/utils/index.ts

import { AxiosResponse } from 'axios';

export const timeRanges: { [key: string]: number } = {
    'ONE_MINUTE': 60,
    'FIVE_MINUTE': 5 * 60,
    'FIFTEEN_MINUTE': 15 * 60,
    'THIRTY_MINUTE': 30 * 60,
    'ONE_HOUR': 60 * 60,
    'TWO_HOUR': 2 * 60 * 60,
    'SIX_HOUR': 6 * 60 * 60,
    'ONE_DAY': 24 * 60 * 60
};

export function getCurrentUnixTimestamp(): number {
    return Math.floor(Date.now() / 1000);
}

export function calculateTimeFrames(granularity: string, endTime: number, candlesPerRequest: number) {
    const timeFrameInSeconds = timeRanges[granularity];
    const timeFrames = [];

    while (endTime > getCurrentUnixTimestamp() - 3 * 30 * 24 * 60 * 60) {
        const startTime = endTime - (candlesPerRequest * timeFrameInSeconds);
        timeFrames.push({ startTime, endTime });
        endTime = startTime;
    }

    return timeFrames.reverse();
}

export function delay(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

export function handleException(response: AxiosResponse) {
    if (response.status !== 200) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
}
