import { listProducts, filterPairings } from './scripts/coinbase/listProducts';
import fetchAndSaveCandles from './scripts/coinbase/getProductCandles';
import { getCurrentUnixTimestamp } from './utils';
import fs from 'fs';
import path from 'path';
import { createObjectCsvWriter } from 'csv-writer';

async function main() {
    const currency = 'USD';
    const granularities = ['ONE_DAY'];
    const startDate = getCurrentUnixTimestamp() - 14 * 24 * 60 * 60; // 14 days ago
    const endDate = getCurrentUnixTimestamp(); // Now

    // Get a list of all pairings with USD currency
    const products = await listProducts();
    const filteredPairings = filterPairings(products, currency);

    // Prepare the output directory
    const outputDir = path.join(__dirname, `../output/pairings-analysis/${startDate}-${endDate}`);
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const results = [];

    for (const pairing of filteredPairings) {
        const [baseCurrency, quoteCurrency] = pairing.split('-');
        const productId = `${baseCurrency}-${quoteCurrency}`;

        for (const granularity of granularities) {
            try {
                const candles = await fetchAndSaveCandles(productId, granularity, startDate, endDate);

                // Analyze the fetched data
                const dailyAnalysis = analyzeDailyCandles(candles);

                // Save individual analysis to file
                const individualFilePath = path.join(outputDir, `${productId}_${granularity}_${startDate}_${endDate}.csv`);
                saveAnalysisToFile(individualFilePath, dailyAnalysis);

                // Summarize the daily analysis to a single total analysis
                const totalAnalysis = summarizeDailyAnalysis(dailyAnalysis);

                results.push({ productId, granularity, ...totalAnalysis });
            } catch (error) {
                console.error(`Failed to fetch candles for ${productId} at ${granularity}:`, error);
            }
        }
    }

    // Save overall analysis results to a CSV file, sorted by volatility
    const overallOutputDir = path.join(__dirname, `../output/pairings-analysis`);
    if (!fs.existsSync(overallOutputDir)) {
        fs.mkdirSync(overallOutputDir, { recursive: true });
    }
    const overallFilePath = path.join(overallOutputDir, `overall_analysis_${startDate}_${endDate}.csv`);
    saveOverallAnalysisToFile(overallFilePath, results);
}

function analyzeDailyCandles(candles: any[]) {
    const dailyAnalysis = [];
    for (const candle of candles) {
        const { start, high, low } = candle;
        const analysis = {
            date: start,
            highestHigh: high,
            lowestLow: low,
            averageSwing: high - low,
            percentageSwingUp: ((high - low) / low) * 100,
            percentageSwingDown: ((high - low) / high) * 100,
        };
        dailyAnalysis.push(analysis);
    }
    return dailyAnalysis;
}

function summarizeDailyAnalysis(dailyAnalysis: any[]) {
    let highestHigh = -Infinity;
    let lowestLow = Infinity;
    let totalSwing = 0;
    let totalDays = dailyAnalysis.length;

    dailyAnalysis.forEach(analysis => {
        highestHigh = Math.max(highestHigh, analysis.highestHigh);
        lowestLow = Math.min(lowestLow, analysis.lowestLow);
        totalSwing += analysis.averageSwing;
    });

    const averageSwing = totalSwing / totalDays;
    const percentageSwingUp = ((highestHigh - lowestLow) / lowestLow) * 100;
    const percentageSwingDown = ((highestHigh - lowestLow) / highestHigh) * 100;

    return {
        highestHigh,
        lowestLow,
        averageSwing,
        percentageSwingUp,
        percentageSwingDown
    };
}

function saveAnalysisToFile(filePath: string, dailyAnalysis: any[]) {
    const totalAnalysis = summarizeDailyAnalysis(dailyAnalysis);

    const csvWriter = createObjectCsvWriter({
        path: filePath,
        header: [
            { id: 'date', title: 'Date' },
            { id: 'highestHigh', title: 'Highest High' },
            { id: 'lowestLow', title: 'Lowest Low' },
            { id: 'averageSwing', title: 'Average Swing' },
            { id: 'percentageSwingUp', title: 'Percentage Swing Up' },
            { id: 'percentageSwingDown', title: 'Percentage Swing Down' }
        ]
    });

    const records = dailyAnalysis.concat({
        date: 'Total',
        ...totalAnalysis
    });

    csvWriter.writeRecords(records).then(() => {
        console.log(`Analysis saved to ${filePath}`);
    });
}

function saveOverallAnalysisToFile(filePath: string, results: any[]) {
    results.sort((a, b) => b.percentageSwingUp - a.percentageSwingUp);

    const csvWriter = createObjectCsvWriter({
        path: filePath,
        header: [
            { id: 'productId', title: 'Product ID' },
            { id: 'granularity', title: 'Granularity' },
            { id: 'highestHigh', title: 'Highest High' },
            { id: 'lowestLow', title: 'Lowest Low' },
            { id: 'averageSwing', title: 'Average Swing' },
            { id: 'percentageSwingUp', title: 'Percentage Swing Up' },
            { id: 'percentageSwingDown', title: 'Percentage Swing Down' }
        ]
    });

    csvWriter.writeRecords(results).then(() => {
        console.log(`Overall analysis saved to ${filePath}`);
    });
}

process.on('SIGINT', () => {
    console.log('\nProcess terminated. Exiting...');
    process.exit(0);
});

main();
