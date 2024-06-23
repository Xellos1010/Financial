const fs = require('fs');
const path = require('path');
const csvParser = require('csv-parser');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

const candlesDirPath = path.join(__dirname, '../../../data/coinbase/candles');

// Function to read and parse a CSV file
async function readCSV(filePath) {
    return new Promise((resolve, reject) => {
        const results = [];
        fs.createReadStream(filePath)
            .pipe(csvParser())
            .on('data', (data) => results.push(data))
            .on('end', () => resolve(results))
            .on('error', (error) => reject(error));
    });
}

// Function to write sorted data back to CSV
async function writeSortedCSV(filePath, data) {
    if (data.length === 0) {
        console.log(`No data to write for ${filePath}. Skipping file.`);
        return;
    }

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

    await csvWriter.writeRecords(data);
    console.log(`CSV file ${filePath} has been sorted and saved.`);
}

// Function to get all CSV files in a directory recursively
function getAllCSVFiles(dir) {
    const files = [];
    fs.readdirSync(dir).forEach(file => {
        const fullPath = path.join(dir, file);
        if (fs.statSync(fullPath).isDirectory()) {
            files.push(...getAllCSVFiles(fullPath));
        } else if (fullPath.endsWith('.csv')) {
            files.push(fullPath);
        }
    });
    return files;
}

// Function to sort and save each CSV file in place
async function sortCSVFilesInPlace() {
    const productDirs = fs.readdirSync(candlesDirPath);

    for (const productId of productDirs) {
        if (productId === 'BTC-USD') {
            const productPath = path.join(candlesDirPath, productId);
            const granularityDirs = fs.readdirSync(productPath);

            for (const granularity of granularityDirs) {
                const granularityPath = path.join(productPath, granularity);
                const csvFiles = getAllCSVFiles(granularityPath);

                for (const file of csvFiles) {
                    try {
                        const data = await readCSV(file);
                        if (data.length > 0) {
                            data.sort((a, b) => parseInt(a.start) - parseInt(b.start));
                            await writeSortedCSV(file, data);
                        } else {
                            console.log(`No data found in ${file}. Skipping sorting.`);
                        }
                    } catch (error) {
                        console.error(`Error processing file ${file}:`, error);
                    }
                }
            }
        }
    }
}

// Execute the function to sort CSV files in place
sortCSVFilesInPlace();
