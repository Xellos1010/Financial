const axios = require('axios');
const fs = require('fs');
const path = require('path');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const csvParser = require('csv-parser');

const candlesDirPath = path.join(__dirname, '../../../data/coinbase/candles');

// Define the time range for each granularity in seconds
const timeRanges = {
    'ONE_MINUTE': 60, // 1 minute
    'FIVE_MINUTE': 5 * 60, // 5 minutes
    'FIFTEEN_MINUTE': 15 * 60, // 15 minutes
    'THIRTY_MINUTE': 30 * 60, // 30 minutes
    'ONE_HOUR': 60 * 60, // 1 hour
    'TWO_HOUR': 2 * 60 * 60, // 2 hours
    'SIX_HOUR': 6 * 60 * 60, // 6 hours
    'ONE_DAY': 24 * 60 * 60 // 1 day
};

// Function to get the current timestamp in Unix format
function getCurrentUnixTimestamp() {
    return Math.floor(Date.now() / 1000);
}

// Function to get the last timestamp from the most recent file
function getLastTimestamp(filePath) {
    const data = fs.readFileSync(filePath, 'utf8');
    const lines = data.trim().split('\n');
    const lastLine = lines[lines.length - 1];
    const lastTimestamp = lastLine.split(',')[0]; // Assuming the timestamp is the first element
    return parseInt(lastTimestamp);
}

// Function to get the most recent file in a directory
function getMostRecentFile(dir) {
    const files = fs.readdirSync(dir).filter(file => file.endsWith('.csv'));
    if (files.length === 0) return null;
    files.sort((a, b) => fs.statSync(path.join(dir, b)).mtime - fs.statSync(path.join(dir, a)).mtime);
    return path.join(dir, files[0]);
}

// Function to calculate start and end times for each request
function calculateTimeFrames(granularity, startTime, endTime, candlesPerRequest) {
    const timeFrameInSeconds = timeRanges[granularity];
    const timeFrames = [];
    
    while (startTime < endTime) {
        const end = startTime + (candlesPerRequest * timeFrameInSeconds);
        timeFrames.push({ startTime, endTime: Math.min(end, endTime) });
        startTime = end;
    }

    return timeFrames;
}

// Function to delay execution
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Function to fetch and save candles data for a given product and granularity
async function fetchAndSaveCandles(productId, granularity, startTime, endTime, retryCount = 0) {
    const maxRetries = 3;
    const url = `https://api.coinbase.com/api/v3/brokerage/market/products/${productId}/candles?start=${startTime}&end=${endTime}&granularity=${granularity}`;
    const productDir = path.join(candlesDirPath, productId, granularity);

    // Ensure product and granularity directories exist
    if (!fs.existsSync(productDir)) {
        fs.mkdirSync(productDir, { recursive: true });
    }

    const filePath = path.join(productDir, `${productId}_${granularity}_${startTime}_${endTime}.csv`);

    // Read existing data from the CSV file, if it exists
    let existingData = [];
    if (fs.existsSync(filePath)) {
        existingData = await new Promise((resolve, reject) => {
            const results = [];
            fs.createReadStream(filePath)
                .pipe(csvParser())
                .on('data', (data) => results.push(data))
                .on('end', () => resolve(results))
                .on('error', (error) => reject(error));
        });
    }

    try {
        const response = await axios.get(url, {
            timeout: 60000, // Set timeout to 60 seconds
        });
        const candles = response.data.candles.map(candle => ({
            start: candle[0],
            low: candle[1],
            high: candle[2],
            open: candle[3],
            close: candle[4],
            volume: candle[5]
        }));

        // Combine existing data with new data and sort by timestamp
        const combinedData = [...existingData, ...candles].sort((a, b) => a.start - b.start);

        const csvWriter = createCsvWriter({
            path: filePath,
            header: [
                { id: 'start', title: 'Start' },
                { id: 'low', title: 'Low' },
                { id: 'high', title: 'High' },
                { id: 'open', title: 'Open' },
                { id: 'close', title: 'Close' },
                { id: 'volume', title: 'Volume' }
            ]
        });

        await csvWriter.writeRecords(combinedData);
        console.log(`Candles for product ${productId} at ${granularity} granularity from ${startTime} to ${endTime} have been written to CSV file successfully.`);
    } catch (error) {
        const curlCommand = `curl --insecure -X GET "${url}" -H "Accept: application/json" -H "Content-Type: application/json"`;
        console.error(`Error fetching candles for product ${productId} at ${granularity} granularity from ${startTime} to ${endTime}:`, error.message, error.response ? error.response.data : '');
        console.log(`Try this command in your terminal to debug:`);
        console.log(curlCommand);
        console.log('Press Ctrl+C to cancel...');
        await delay(5000); // Wait 5 seconds to allow for manual inspection
    }
}

// Function to fetch candles data for BTC-USD product
async function fetchCandlesData() {
    const productId = 'BTC-USD';
    const granularities = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'TWO_HOUR', 'SIX_HOUR', 'ONE_DAY'];

    for (const granularity of granularities) {
        const candlesPerRequest = 300;
        const productDir = path.join(candlesDirPath, productId, granularity);
        const mostRecentFile = getMostRecentFile(productDir);
        let startTime = mostRecentFile ? getLastTimestamp(mostRecentFile) + timeRanges[granularity] : getCurrentUnixTimestamp() - 3 * 30 * 24 * 60 * 60; // 3 months ago if no file exists
        const endTime = getCurrentUnixTimestamp();

        const timeFrames = calculateTimeFrames(granularity, startTime, endTime, candlesPerRequest);

        for (let i = 0; i < timeFrames.length; i++) {
            if (i > 0 && i % 10 === 0) { // Rate limit: 10 requests per second
                await delay(1000); // Wait 1 second
            }
            const { startTime, endTime } = timeFrames[i];
            await fetchAndSaveCandles(productId, granularity, startTime, endTime);
        }
    }
}

// Handle Ctrl+C for graceful shutdown
process.on('SIGINT', () => {
    console.log('\nProcess terminated. Exiting...');
    process.exit(0);
});

// Execute the function to fetch candles data
fetchCandlesData();
