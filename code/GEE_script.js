// 1. Define Study Area (Bokaro, Jharkhand)
var aoi = ee.FeatureCollection("FAO/GAUL/2015/level2")
  .filter(ee.Filter.eq('ADM2_NAME', 'Bokaro'));
Map.centerObject(aoi, 10);
Map.addLayer(aoi, {color: 'red'}, 'Bokaro Boundary', false);

// 2. Cloud Masking Function for Landsat 8/9
function maskL8sr(image) {
  var qa = image.select('QA_PIXEL');
  var cloudShadowBitMask = (1 << 4);
  var cloudsBitMask = (1 << 3);
  
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
    .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
    
  return image.updateMask(mask).clip(aoi);
}

function addBands(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBand = image.select('ST_B10').multiply(0.00341802).add(149.0);
  
  image = image.addBands(opticalBands, null, true).addBands(thermalBand, null, true);

  var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  var ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI');
  var mndwi = image.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI');
  
  image = image.addBands([ndvi, ndbi, mndwi]);

  var fvc = image.expression('((NDVI - 0.1) / (0.5)) ** 2', {'NDVI': ndvi}).rename('FVC');
  var emissivity = fvc.multiply(0.004).add(0.986).rename('LSE');
  
  var lst = image.expression(
    '(BT / (1 + (0.00115 * BT / 1.4388) * log(LSE))) - 273.15', {
      'BT': image.select('ST_B10'), 'LSE': emissivity
    }).rename('LST_Celsius');

  return image.addBands(lst);
}

// Fetch 2015 and 2025 imagery with STRICT scene-level cloud filtering
var img2015 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filterBounds(aoi)
  .filterDate('2015-03-01', '2015-05-31') 
  .filter(ee.Filter.lt('CLOUD_COVER', 15)) 
  .map(maskL8sr)
  .map(addBands)
  .median(); 

// Note: Using 2024 dates based on your previous adjustments
var img2025 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filterBounds(aoi)
  .filterDate('2024-03-01', '2024-05-31') 
  .filter(ee.Filter.lt('CLOUD_COVER', 15)) 
  .map(maskL8sr)
  .map(addBands)
  .median();

Map.addLayer(img2015, {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0.0, max: 0.3}, 'True Color 2015', false);
Map.addLayer(img2025, {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0.0, max: 0.3}, 'True Color 2025', false);

// Features to predict on (Strictly spectral and indices)
var predictionBands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'NDVI', 'NDBI', 'MNDWI'];

// LOAD TRAINING DATA DIRECTLY FROM ASSETS
var trainingPoints15 = ee.FeatureCollection('users/ayushman/bokaro_training_2015');
var trainingPoints25 = ee.FeatureCollection('users/ayushman/bokaro_training_2024');

// ==========================================
// --- Train & Validate the 2015 Classifier ---
// (With Spatially Independent Polygon Split)
// ==========================================

// 1. Add random column to the POLYGONS (using seed 42 for stable stratification)
var polygonsWithRandom15 = trainingPoints15.randomColumn('random', 42);
var split = 0.8;

// 2. Split polygons into Train (80%) and Test (20%)
var trainingPolygons15 = polygonsWithRandom15.filter(ee.Filter.lt('random', split));
var testingPolygons15 = polygonsWithRandom15.filter(ee.Filter.gte('random', split));

// 3. Extract pixels ONLY from training polygons
var trainingData15 = img2015.select(predictionBands).sampleRegions({
  collection: trainingPolygons15, 
  properties: ['landcover'], 
  scale: 30,
  tileScale: 16 // Prevents memory limits during extraction
});

// 4. Extract pixels ONLY from testing polygons
var testingData15 = img2015.select(predictionBands).sampleRegions({
  collection: testingPolygons15, 
  properties: ['landcover'], 
  scale: 30,
  tileScale: 16
});

var classifier15 = ee.Classifier.smileRandomForest({
  numberOfTrees: 50, 
  seed: 42
}).train({
  features: trainingData15, classProperty: 'landcover', inputProperties: predictionBands
});

// 2015 Accuracy Assessment (Spatially Independent)
var validated15 = testingData15.classify(classifier15);
var errorMatrix15 = validated15.errorMatrix('landcover', 'classification');

print('--- 2015 Validation Metrics (Spatially Independent) ---');
print('2015 Confusion Matrix:', errorMatrix15);
print('2015 Overall Accuracy:', errorMatrix15.accuracy());
print('2015 Kappa Coefficient:', errorMatrix15.kappa());
print('2015 Producers Accuracy (Recall):', errorMatrix15.producersAccuracy());
print('2015 Users Accuracy (Precision):', errorMatrix15.consumersAccuracy());

var classified2015 = img2015.select(predictionBands).classify(classifier15); 

// ==========================================
// --- Train & Validate the 2025 Classifier ---
// (With Spatially Independent Polygon Split)
// ==========================================

// 1. Add random column to the POLYGONS (using seed 42)
var polygonsWithRandom25 = trainingPoints25.randomColumn('random', 42);

// 2. Split polygons into Train (80%) and Test (20%)
var trainingPolygons25 = polygonsWithRandom25.filter(ee.Filter.lt('random', split));
var testingPolygons25 = polygonsWithRandom25.filter(ee.Filter.gte('random', split));

// 3. Extract pixels ONLY from training polygons
var trainingData25 = img2025.select(predictionBands).sampleRegions({
  collection: trainingPolygons25, 
  properties: ['landcover'], 
  scale: 30,
  tileScale: 16
});

// 4. Extract pixels ONLY from testing polygons
var testingData25 = img2025.select(predictionBands).sampleRegions({
  collection: testingPolygons25, 
  properties: ['landcover'], 
  scale: 30,
  tileScale: 16
});

var classifier25 = ee.Classifier.smileRandomForest({
  numberOfTrees: 50, 
  seed: 42
}).train({
  features: trainingData25, classProperty: 'landcover', inputProperties: predictionBands
});

// 2025 Accuracy Assessment (Spatially Independent)
var validated25 = testingData25.classify(classifier25);
var errorMatrix25 = validated25.errorMatrix('landcover', 'classification');

print('--- 2025 Validation Metrics (Spatially Independent) ---');
print('2025 Confusion Matrix:', errorMatrix25);
print('2025 Overall Accuracy:', errorMatrix25.accuracy());
print('2025 Kappa Coefficient:', errorMatrix25.kappa());
print('2025 Producers Accuracy (Recall):', errorMatrix25.producersAccuracy());
print('2025 Users Accuracy (Precision):', errorMatrix25.consumersAccuracy());

var classified2025 = img2025.select(predictionBands).classify(classifier25);

// Visualize Classifications
var visParams = {min: 0, max: 3, palette: ['red', 'green', 'yellow', 'blue']};
Map.addLayer(classified2015, visParams, 'LULC 2015');
Map.addLayer(classified2025, visParams, 'LULC 2025', false);


// --- 2015 Metrics ---
var meanLst2015 = img2015.select('LST_Celsius').reduceRegion({
  reducer: ee.Reducer.mean(), geometry: aoi, scale: 30, maxPixels: 1e9
});
print('Mean LST 2015 (°C):', meanLst2015.get('LST_Celsius'));

var builtUpArea2015 = ee.Image.pixelArea().updateMask(classified2015.eq(0)).reduceRegion({
  reducer: ee.Reducer.sum(), geometry: aoi, scale: 30, maxPixels: 1e9
});
var builtUpSqKm2015 = ee.Number(builtUpArea2015.get('area')).divide(1e6);
print('Built-up Area 2015 (sq km):', builtUpSqKm2015);

// --- 2025 Metrics ---
var meanLst2025 = img2025.select('LST_Celsius').reduceRegion({
  reducer: ee.Reducer.mean(), geometry: aoi, scale: 30, maxPixels: 1e9
});
print('Mean LST 2024 (°C):', meanLst2025.get('LST_Celsius'));

var builtUpArea2025 = ee.Image.pixelArea().updateMask(classified2025.eq(0)).reduceRegion({
  reducer: ee.Reducer.sum(), geometry: aoi, scale: 30, maxPixels: 1e9
});
var builtUpSqKm2025 = ee.Number(builtUpArea2025.get('area')).divide(1e6);
print('Built-up Area 2024 (sq km):', builtUpSqKm2025);

// Thermal Layer for Visualization Grid
Map.addLayer(img2015.select('LST_Celsius'), {min: 35, max: 50, palette: ['blue', 'yellow', 'red']}, 'Thermal 2015', false);
Map.addLayer(img2025.select('LST_Celsius'), {min: 35, max: 50, palette: ['blue', 'yellow', 'red']}, 'Thermal 2025', false);

// Calculate the exact temperature change per pixel over 10 years
var lstDifference = img2025.select('LST_Celsius').subtract(img2015.select('LST_Celsius'));

//Isolate the New Urban Expansion (Built-up in 2025, but NOT in 2015)
var newUrbanExpansion = classified2025.eq(0).and(classified2015.neq(0));

// Mask the zeros so only the expanded pixels render on the map
var expansionMasked = newUrbanExpansion.updateMask(newUrbanExpansion);

// Add to map in a high-contrast color (Magenta) so it pops over the thermal gradient
Map.addLayer(expansionMasked, {palette: ['magenta']}, 'New Urban Expansion (2015-2025)', false);

// Visualization parameters for the thermal delta
var deltaVis = {min: -2, max: 6, palette: ['blue', 'white', 'orange', 'red', 'darkred']};
Map.addLayer(lstDifference, deltaVis, '10-Year LST Change (°C)', false);

// Create a styling panel to hold the legend
var legend = ui.Panel({
  style: { position: 'bottom-right', padding: '8px 15px', backgroundColor: 'rgba(255, 255, 255, 0.9)' }
});

// Create a title for the legend
var legendTitle = ui.Label({
  value: '10-Yr LST Change (°C)',
  style: { fontWeight: 'bold', fontSize: '14px', margin: '0 0 4px 0', padding: '0' }
});
legend.add(legendTitle);

// Function to generate the gradient color bar
var makeColorBarParams = function(palette) {
  return {
    bbox: [0, 0, 1, 0.1],
    dimensions: '100x10',
    format: 'png',
    min: 0,
    max: 1,
    palette: palette,
  };
};

// Add the gradient color bar to the legend
var colorBar = ui.Thumbnail({
  image: ee.Image.pixelLonLat().select(0),
  params: makeColorBarParams(deltaVis.palette),
  style: { stretch: 'horizontal', margin: '0px 8px', maxHeight: '20px' },
});
legend.add(colorBar);

// Create numeric labels for the color bar
var legendLabels = ui.Panel({
  widgets: [
    ui.Label(deltaVis.min, {margin: '4px 8px'}),
    ui.Label(((deltaVis.max + deltaVis.min) / 2), {margin: '4px 8px', textAlign: 'center', stretch: 'horizontal'}),
    ui.Label(deltaVis.max, {margin: '4px 8px'})
  ],
  layout: ui.Panel.Layout.flow('horizontal')
});
legend.add(legendLabels);

Map.add(legend);

Export.table.toDrive({
  // Note: Only exporting the training partition here to match your Python workflow
  collection: trainingData25,
  description: 'Bokaro_2025_LULC_Features',
  fileFormat: 'CSV'
});

// ==========================================
// GEOTIFF EXPORT FOR PYTHON INFERENCE
// ==========================================

// Ensure the exact band order matching your CSV
var predictionBands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'NDVI', 'NDBI', 'MNDWI'];

// Force all bands to have the exact same data type (Float32) to prevent Export Error 3
var exportImage = img2025.select(predictionBands).toFloat();

Export.image.toDrive({
  image: exportImage, 
  description: 'Bokaro_2025_GeoTIFF',
  folder: 'EarthEngine_Exports', 
  fileNamePrefix: 'Bokaro_2025_Features',
  region: aoi,
  scale: 30, // 30m Landsat resolution
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Isolate stable rural/vegetated pixels (Class 1 in both years)
var stableRuralMask = classified2015.eq(1).and(classified2025.eq(1));

// Calculate Mean LST for the Rural Control in 2015
var ruralLst2015 = img2015.select('LST_Celsius').updateMask(stableRuralMask).reduceRegion({
  reducer: ee.Reducer.mean(), geometry: aoi, scale: 30, maxPixels: 1e9
});

// Calculate Mean LST for the Rural Control in 2025
var ruralLst2025 = img2025.select('LST_Celsius').updateMask(stableRuralMask).reduceRegion({
  reducer: ee.Reducer.mean(), geometry: aoi, scale: 30, maxPixels: 1e9
});

print('Stable Rural LST 2015:', ruralLst2015.get('LST_Celsius'));
print('Stable Rural LST 2016:', ruralLst2025.get('LST_Celsius'));

// Fetch ERA5-Land Monthly Aggregated Data
var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterBounds(aoi);

// Get Mean 2m Air Temperature for 2015 Window (Convert Kelvin to Celsius)
var airTemp2015 = era5.filterDate('2015-03-01', '2015-05-31')
  .select('temperature_2m').mean().subtract(273.15)
  .reduceRegion({reducer: ee.Reducer.mean(), geometry: aoi, scale: 9000});

// Get Mean 2m Air Temperature for 2025 Window
var airTemp2025 = era5.filterDate('2024-03-01', '2024-05-31')
  .select('temperature_2m').mean().subtract(273.15)
  .reduceRegion({reducer: ee.Reducer.mean(), geometry: aoi, scale: 9000});

print('ERA5 Regional Air Temp 2015 (°C):', airTemp2015.get('temperature_2m'));
print('ERA5 Regional Air Temp 2016 (°C):', airTemp2025.get('temperature_2m'));

// Combine Mean and Standard Deviation reducers
var combinedReducer = ee.Reducer.mean().combine({
  reducer2: ee.Reducer.stdDev(),
  sharedInputs: true
});

// --- Calculate stats for 2015 LST ---
var stats2015 = img2015.select('LST_Celsius').reduceRegion({
  reducer: combinedReducer,
  geometry: aoi,
  scale: 30,
  maxPixels: 1e9
});

print('2015 LST Mean (°C):', stats2015.get('LST_Celsius_mean'));
print('2015 LST StdDev (Uncertainty):', stats2015.get('LST_Celsius_stdDev'));

// --- Calculate stats for 2024 LST ---
var stats2025 = img2025.select('LST_Celsius').reduceRegion({
  reducer: combinedReducer,
  geometry: aoi,
  scale: 30,
  maxPixels: 1e9
});

print('2024 LST Mean (°C):', stats2025.get('LST_Celsius_mean'));
print('2024 LST StdDev (Uncertainty):', stats2025.get('LST_Celsius_stdDev'));

// Load your 2015 training asset
var trainingPoints15 = ee.FeatureCollection('users/ayushman/bokaro_training_2015');

// Calculate the frequency of each class in the 'landcover' column
var classCounts15 = trainingPoints15.aggregate_histogram('landcover');

// Print the counts to the console
print('2015 Polygon Counts per Class:', classCounts15);

// Calculate the total number of polygons drawn
var totalPolygons15 = trainingPoints15.size();
print('Total Polygons Drawn (2015):', totalPolygons15);
