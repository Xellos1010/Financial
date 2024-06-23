// src/scripts/coinbase/getProductCandles.ts
import axios from 'axios';
import { delay, handleException } from '../../utils';

async function fetchAndSaveCandles(productId: string, granularity: string, startTime: number, endTime: number) {
    const url = `https://api.coinbase.com/api/v3/brokerage/market/products/${productId}/candles?start=${startTime}&end=${endTime}&granularity=${granularity}`;

    const maxRetries = 5;
    const initialBackoff = 1000;
    let attempts = 0;

    while (attempts < maxRetries) {
        try {
            const response = await axios.get(url, { timeout: 60000 });
            handleException(response);
            const candles = response.data.candles;

            console.log(`Candles for product ${productId} at ${granularity} granularity from ${startTime} to ${endTime} have been fetched successfully.`);
            return candles;
        } catch (error) {
            if (axios.isAxiosError(error) && error.response?.status === 429) {
                attempts++;
                const backoffTime = initialBackoff * Math.pow(2, attempts);
                console.log(`Rate limited. Retrying in ${backoffTime}ms... (Attempt ${attempts}/${maxRetries})`);
                await delay(backoffTime);
            } else {
                throw error;
            }
        }
    }
    throw new Error(`Max retries exceeded for product ${productId} at ${granularity} granularity from ${startTime} to ${endTime}`);
}

export default fetchAndSaveCandles;
