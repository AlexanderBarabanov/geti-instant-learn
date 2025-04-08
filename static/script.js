document.addEventListener('DOMContentLoaded', () => {
    const processButton = document.getElementById('process-button');
    const pipelineSelect = document.getElementById('pipeline');
    const datasetSelect = document.getElementById('dataset');
    const classNameSelect = document.getElementById('class_name');
    const nShotInput = document.getElementById('n_shot');
    const statusSpan = document.getElementById('status');
    const resultsContainer = document.getElementById('results-container');

    // Store image objects, mask data, point data, and clickable regions
    const canvasDataStore = {};
    // { canvasId: {
    //      image: Image,
    //      masks: [{..., mask_data_uri}],
    //      points: [],
    //      element: HTMLCanvasElement,
    //      clickablePoints: [] // Store {x, y, radius, maskInstanceId} here
    //   }
    // }
    const maskImageCache = {};

    // Define some colors for point visualization (reuse similar to python)
    const COLORS = [
        'rgba(255, 0, 0, 1)', 'rgba(0, 255, 0, 1)', 'rgba(0, 0, 255, 1)',
        'rgba(255, 255, 0, 1)', 'rgba(0, 255, 255, 1)', 'rgba(255, 0, 255, 1)',
        'rgba(128, 0, 0, 1)', 'rgba(0, 128, 0, 1)', 'rgba(0, 0, 128, 1)',
        'rgba(128, 128, 0, 1)', 'rgba(0, 128, 128, 1)', 'rgba(128, 0, 128, 1)'
    ];

    // Function to fetch classes for the selected dataset
    async function fetchAndPopulateClasses (selectedDataset) {
        statusSpan.textContent = 'Loading classes...';
        statusSpan.classList.add('loading');
        classNameSelect.innerHTML = '<option value=\"\" disabled selected>Loading...</option>'; // Clear existing options
        try {
            const response = await fetch(`/api/classes?dataset=${encodeURIComponent(selectedDataset)}`);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            classNameSelect.innerHTML = '<option value=\"\" disabled selected>Select Class...</option>'; // Reset placeholder
            if (data.classes && data.classes.length > 0) {
                data.classes.forEach(className => {
                    const option = document.createElement('option');
                    option.value = className;
                    option.textContent = className;
                    classNameSelect.appendChild(option);
                });
            } else {
                classNameSelect.innerHTML = '<option value=\"\" disabled selected>No classes found</option>';
            }
            statusSpan.textContent = ''; // Clear status
            statusSpan.classList.remove('loading');
        } catch (error) {
            console.error('Error fetching classes:', error);
            statusSpan.textContent = `Error loading classes: ${error.message}`;
            statusSpan.style.color = 'red';
            classNameSelect.innerHTML = '<option value=\"\" disabled selected>Error loading</option>';
            statusSpan.classList.remove('loading');
        }
    }

    // Event listener for dataset change
    datasetSelect.addEventListener('change', () => {
        fetchAndPopulateClasses(datasetSelect.value);
    });

    // Initial population of classes on page load
    fetchAndPopulateClasses(datasetSelect.value);

    processButton.addEventListener('click', async () => {
        const pipeline = pipelineSelect.value;
        const dataset = datasetSelect.value;
        const className = classNameSelect.value;
        const nShot = parseInt(nShotInput.value, 10);

        if (!className) {
            statusSpan.textContent = 'Error: Please select a Class Name.';
            statusSpan.style.color = 'red';
            return;
        }
        if (isNaN(nShot) || nShot < 1) {
            statusSpan.textContent = 'Error: N-Shot must be a number >= 1.';
            statusSpan.style.color = 'red';
            return;
        }

        statusSpan.textContent = 'Processing...';
        statusSpan.style.color = 'black';
        statusSpan.classList.add('loading');
        resultsContainer.innerHTML = ''; // Clear previous results
        Object.keys(canvasDataStore).forEach(key => delete canvasDataStore[key]); // Clear data store
        // Clear mask cache for new run? Or keep globally?
        // Let's clear it per run for simplicity now.
        Object.keys(maskImageCache).forEach(key => delete maskImageCache[key]);

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    pipeline: pipeline,
                    dataset: dataset,
                    class_name: className,
                    n_shot: nShot,
                }),
            });

            statusSpan.classList.remove('loading');

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (!data.target_results || data.target_results.length === 0) {
                statusSpan.textContent = 'No target results found.';
                return;
            }

            statusSpan.textContent = 'Processing complete.';
            displayResults(data.target_results);

        } catch (error) {
            console.error('Error fetching or processing data:', error);
            statusSpan.textContent = `Error: ${error.message}`;
            statusSpan.style.color = 'red';
            statusSpan.classList.remove('loading');
        }
    });

    function displayResults (targetResults) {
        targetResults.forEach((result, index) => {
            const targetItemDiv = document.createElement('div');
            targetItemDiv.classList.add('target-item');

            const canvasId = `canvas-${index}`;
            const canvas = document.createElement('canvas');
            canvas.id = canvasId;

            const maskControlsDiv = document.createElement('div');
            maskControlsDiv.classList.add('mask-controls');
            maskControlsDiv.innerHTML = '<strong>Masks:</strong>'; // Start with title

            // Add Select All button
            const selectAllButton = document.createElement('button');
            selectAllButton.textContent = 'Select All';
            selectAllButton.style.marginLeft = '10px';
            selectAllButton.style.fontSize = '0.8em';
            selectAllButton.dataset.canvasId = canvasId;
            selectAllButton.addEventListener('click', (event) => {
                const targetCanvasId = event.target.dataset.canvasId;
                const controlsDiv = event.target.parentElement;
                const checkboxes = controlsDiv.querySelectorAll('input[type="checkbox"]');
                checkboxes.forEach(cb => cb.checked = true); // Set to true
                redrawCanvas(targetCanvasId);
            });
            maskControlsDiv.appendChild(selectAllButton);

            // Add Unselect All button
            const unselectAllButton = document.createElement('button');
            unselectAllButton.textContent = 'Unselect All';
            unselectAllButton.style.marginLeft = '10px'; // Add some spacing
            unselectAllButton.style.fontSize = '0.8em';
            unselectAllButton.dataset.canvasId = canvasId; // Link button to canvas
            unselectAllButton.addEventListener('click', (event) => {
                const targetCanvasId = event.target.dataset.canvasId;
                const controlsDiv = event.target.parentElement;
                const checkboxes = controlsDiv.querySelectorAll('input[type="checkbox"]');
                checkboxes.forEach(cb => cb.checked = false);
                redrawCanvas(targetCanvasId); // Redraw after unselecting
            });
            maskControlsDiv.appendChild(unselectAllButton);
            maskControlsDiv.appendChild(document.createElement('br')); // Add line break

            targetItemDiv.appendChild(canvas);
            targetItemDiv.appendChild(maskControlsDiv);
            resultsContainer.appendChild(targetItemDiv);

            // Store data needed for redraws
            canvasDataStore[canvasId] = {
                image: null,
                masks: result.masks || [],
                points: result.points || [],
                element: canvas,
                clickablePoints: [] // Initialize clickable points storage
            };

            // Add click listener to the canvas
            canvas.addEventListener('click', handleCanvasClick);

            // Create checkboxes and add listeners
            if (result.masks && result.masks.length > 0) {
                result.masks.forEach(mask => {
                    const checkboxId = `${canvasId}-mask-${mask.instance_id}`;
                    const label = document.createElement('label');
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.id = checkboxId;
                    checkbox.value = mask.instance_id;
                    checkbox.checked = true; // Default to visible
                    checkbox.dataset.canvasId = canvasId; // Link checkbox to canvas

                    label.appendChild(checkbox);
                    label.appendChild(document.createTextNode(` ${mask.instance_id} (Class ${mask.class_id})`));
                    maskControlsDiv.appendChild(label);

                    // Add listener to redraw when checkbox changes
                    checkbox.addEventListener('change', (event) => {
                        const targetCanvasId = event.target.dataset.canvasId;
                        redrawCanvas(targetCanvasId);
                    });
                });
            } else {
                maskControlsDiv.innerHTML += '<i>No masks found</i>';
            }

            // Preload mask images using data URIs
            preloadMaskImages(canvasId);

            // Load base image onto canvas using data URI
            const img = new Image();
            img.onload = () => {
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                canvasDataStore[canvasId].image = img;
                redrawCanvas(canvasId);
            };
            img.onerror = () => {
                console.error(`Failed to load image from data URI`);
                const ctx = canvas.getContext('2d');
                // Draw error text on canvas if image fails to load
                canvas.width = 300; // Example dimensions
                canvas.height = 100;
                ctx.fillStyle = 'red';
                ctx.font = '16px sans-serif';
                ctx.fillText('Error loading image', 10, 50);
            };
            img.src = result.image_data_uri; // Use data URI directly
        });
    }

    function preloadMaskImages (canvasId) {
        const data = canvasDataStore[canvasId];
        if (!data || !data.masks) return;

        data.masks.forEach(mask => {
            const uri = mask.mask_data_uri;
            if (uri && !maskImageCache[uri]) { // Check if URI exists
                const maskImg = new Image();
                maskImg.onload = () => {
                    maskImageCache[uri] = maskImg; // Store loaded image in cache
                };
                maskImg.onerror = () => {
                    console.error(`Failed to preload mask image from data URI`);
                    maskImageCache[uri] = null; // Mark as failed
                };
                maskImageCache[uri] = 'loading'; // Mark as loading
                maskImg.src = uri; // Use data URI directly
            }
        });
    }

    function drawStar (ctx, x, y, outerRadius, innerRadius, points) {
        ctx.beginPath();
        ctx.moveTo(x, y - outerRadius); // Start at top point
        for (let i = 0; i < points; i++) {
            // Outer point
            let angle = Math.PI / points * (2 * i + 1.5); // Angle adjustment to start at top
            ctx.lineTo(x + outerRadius * Math.cos(angle), y + outerRadius * Math.sin(angle));
            // Inner point
            angle = Math.PI / points * (2 * i + 2.5);
            ctx.lineTo(x + innerRadius * Math.cos(angle), y + innerRadius * Math.sin(angle));
        }
        ctx.closePath();
    }

    function redrawCanvas (canvasId) {
        const data = canvasDataStore[canvasId];
        if (!data || !data.image) {
            console.warn(`Data or image not ready for canvas ${canvasId}`);
            return; // Image not loaded yet or data missing
        }

        const canvas = data.element;
        const ctx = canvas.getContext('2d');
        const img = data.image;
        const masks = data.masks;
        const points = data.points;

        // Clear previous clickable points for this canvas
        data.clickablePoints = [];

        // 1. Clear canvas and draw base image
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1.0; // Ensure base image is fully opaque
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // 2. Draw visible mask overlays
        masks.forEach(mask => {
            const checkboxId = `${canvasId}-mask-${mask.instance_id}`;
            const checkbox = document.getElementById(checkboxId);
            const maskUri = mask.mask_data_uri; // Use data URI

            if (checkbox && checkbox.checked && maskUri) {
                const cachedMaskImg = maskImageCache[maskUri]; // Use data URI as key

                if (cachedMaskImg && cachedMaskImg !== 'loading') {
                    ctx.globalAlpha = 0.3; // Alpha for the mask overlay
                    ctx.drawImage(cachedMaskImg, 0, 0, canvas.width, canvas.height);
                    ctx.globalAlpha = 1.0; // Reset alpha
                } else if (cachedMaskImg === 'loading') {
                    // console.log(`Mask from data URI still loading...`);
                } else if (cachedMaskImg === null) {
                    // console.log(`Mask from data URI failed to load.`);
                } else {
                    console.warn(`Mask image not found in cache for data URI`);
                    // preloadMaskImages(canvasId); // Could retry loading
                }
            }
        });

        // 3. Draw points on top AND store clickable regions for FG points
        if (points && points.length > 0) {
            points.forEach((point, idx) => {
                const classId = point.class_id || 0;
                const color = COLORS[classId % COLORS.length];
                const pointX = point.x;
                const pointY = point.y;
                const label = point.label;
                const correspondingMaskInstanceId = `mask_${idx}`; // Assumed link

                // Determine if the mask is selected (for styling and click logic)
                const checkboxId = `${canvasId}-mask-${correspondingMaskInstanceId}`;
                const checkbox = document.getElementById(checkboxId);
                const isMaskSelected = checkbox ? checkbox.checked : false;

                let clickRadius = 0; // Radius for click detection

                if (isMaskSelected) { // Draw prominent star if mask is selected
                    const outerRadius = 30; // Increased outer radius further
                    const innerRadius = 12;  // Increased inner radius further
                    const numPoints = 5;
                    clickRadius = outerRadius;
                    ctx.fillStyle = color;
                    drawStar(ctx, pointX, pointY, outerRadius, innerRadius, numPoints);
                    ctx.fill();
                    ctx.strokeStyle = 'black';
                    ctx.lineWidth = 1;
                    drawStar(ctx, pointX, pointY, outerRadius, innerRadius, numPoints);
                    ctx.stroke();
                    // Store clickable info
                    data.clickablePoints.push({ x: pointX, y: pointY, radius: clickRadius, maskInstanceId: correspondingMaskInstanceId });

                } else { // Mask not selected
                    if (label > 0) { // Foreground point (mask not selected)
                        const radius = 12; // Make unselected FG points larger dots
                        clickRadius = radius;
                        ctx.fillStyle = color;
                        ctx.beginPath();
                        ctx.arc(pointX, pointY, radius, 0, 2 * Math.PI, false);
                        ctx.fill();
                        // Store clickable info
                        data.clickablePoints.push({ x: pointX, y: pointY, radius: clickRadius, maskInstanceId: correspondingMaskInstanceId });

                    } else { // Background point (mask never selected)
                        const rectSize = 20; // Increased rectangle size
                        clickRadius = rectSize / 2;
                        const bgColor = 'rgba(0, 128, 128, 1)';
                        ctx.fillStyle = bgColor;
                        ctx.fillRect(pointX - rectSize / 2, pointY - rectSize / 2, rectSize, rectSize);
                        ctx.strokeStyle = 'black';
                        ctx.lineWidth = 1;
                        ctx.strokeRect(pointX - rectSize / 2, pointY - rectSize / 2, rectSize, rectSize);
                        // Store clickable info for background points too (though they won't toggle masks)
                        // We could potentially add other interactions later
                        data.clickablePoints.push({ x: pointX, y: pointY, radius: clickRadius, maskInstanceId: null }); // No corresponding mask to toggle
                    }
                }
            });
        }
    }

    // Function to handle clicks on the canvas
    function handleCanvasClick (event) {
        const canvas = event.target;
        const canvasId = canvas.id;
        const data = canvasDataStore[canvasId];
        if (!data || !data.clickablePoints) return;

        // Get click coordinates relative to the canvas
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;    // Handle CSS scaling
        const scaleY = canvas.height / rect.height;
        const clickX = (event.clientX - rect.left) * scaleX;
        const clickY = (event.clientY - rect.top) * scaleY;

        let clickedOnPoint = false;

        // Check if the click hit any stored foreground point areas
        // Iterate in reverse to prioritize points drawn on top if overlap occurs
        for (let i = data.clickablePoints.length - 1; i >= 0; i--) {
            const pt = data.clickablePoints[i];
            const distance = Math.sqrt(Math.pow(clickX - pt.x, 2) + Math.pow(clickY - pt.y, 2));

            if (distance <= pt.radius && pt.maskInstanceId) { // Check distance and if it's a FG point with a mask ID
                clickedOnPoint = true;
                const checkboxId = `${canvasId}-mask-${pt.maskInstanceId}`;
                const checkbox = document.getElementById(checkboxId);
                if (checkbox) {
                    checkbox.checked = !checkbox.checked; // Toggle the checkbox
                    redrawCanvas(canvasId); // Redraw to reflect the change
                }
                break; // Stop after finding the first hit (topmost)
            }
        }
        // Optional: Add logic here if click didn't hit any point
        // if (!clickedOnPoint) { console.log("Clicked on background"); }
    }
});
