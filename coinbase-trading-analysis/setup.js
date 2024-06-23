const fs = require('fs');
const path = require('path');

const directories = [
  'data/coinbase/products',
  'data/coinbase/candles',
  'src/scripts/coinbase',
  'src/utils'
];

const files = [
  'src/scripts/coinbase/listProducts.ts',
  'src/scripts/coinbase/getProductCandles.ts',
  'src/utils/index.ts',
  'package.json',
  'tsconfig.json',
  '.gitignore'
];

function createDirectories() {
  directories.forEach(dir => {
    const dirPath = path.join(__dirname, dir);
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
      console.log(`Directory created: ${dir}`);
    } else {
      console.log(`Directory already exists: ${dir}`);
    }
  });
}

function createFiles() {
  files.forEach(file => {
    const filePath = path.join(__dirname, file);
    if (!fs.existsSync(filePath)) {
      fs.writeFileSync(filePath, '');
      console.log(`File created: ${file}`);
    } else {
      console.log(`File already exists: ${file}`);
    }
  });
}

function setup() {
  createDirectories();
  createFiles();
  console.log('Setup completed successfully.');
}

setup();
