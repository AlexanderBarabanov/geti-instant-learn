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

            // --- Create Controls Container ---
            const controlsContainer = document.createElement('div');
            controlsContainer.classList.add('item-controls'); // Add a class for styling

            // --- Point Display Mode Controls ---
            const pointModeDiv = document.createElement('div');
            pointModeDiv.style.marginBottom = '10px';
            pointModeDiv.innerHTML = '<strong>Show Points:</strong> ';

            const modeGroupName = `point-mode-${canvasId}`;

            const radioUsed = document.createElement('input');
            radioUsed.type = 'radio';
            radioUsed.name = modeGroupName;
            radioUsed.id = `mode-used-${canvasId}`;
            radioUsed.value = 'used';
            radioUsed.checked = true; // Default to showing used points
            radioUsed.dataset.canvasId = canvasId;
            radioUsed.addEventListener('change', () => redrawCanvas(canvasId));

            const labelUsed = document.createElement('label');
            labelUsed.htmlFor = `mode-used-${canvasId}`;
            labelUsed.textContent = 'Used Points';
            labelUsed.style.marginRight = '10px';

            const radioAll = document.createElement('input');
            radioAll.type = 'radio';
            radioAll.name = modeGroupName;
            radioAll.id = `mode-all-${canvasId}`;
            radioAll.value = 'all';
            radioAll.dataset.canvasId = canvasId;
            radioAll.addEventListener('change', () => redrawCanvas(canvasId));

            const labelAll = document.createElement('label');
            labelAll.htmlFor = `mode-all-${canvasId}`;
            labelAll.textContent = 'All Points';

            pointModeDiv.appendChild(radioUsed);
            pointModeDiv.appendChild(labelUsed);
            pointModeDiv.appendChild(radioAll);
            pointModeDiv.appendChild(labelAll);
            controlsContainer.appendChild(pointModeDiv);

            // --- Mask Controls (Existing Logic) ---
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
            controlsContainer.appendChild(maskControlsDiv);
            // --- End Mask Controls ---

            targetItemDiv.appendChild(canvas);
            targetItemDiv.appendChild(controlsContainer); // Add the combined controls
            resultsContainer.appendChild(targetItemDiv);

            // Store data needed for redraws
            canvasDataStore[canvasId] = {
                image: null,
                masks: result.masks || [],
                // Store both sets of points
                used_points: result.used_points || [],
                prior_points: result.prior_points || [],
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
            // console.warn(`No image data or image not loaded for canvas ${canvasId}`);
            return; // Don't draw if image isn't ready
        }

        const canvas = data.element;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Calculate a base size for points proportional to image size
        // Use a small fraction of the average dimension, with a minimum size
        const baseSize = Math.max(3, Math.min(canvas.width, canvas.height) * 0.01);

        // Define sizes based on baseSize (make them larger)
        const starOuterRadius = baseSize * 1.8;
        const starInnerRadius = baseSize * 0.7;
        const dotRadius = baseSize * 0.8;
        const squareSize = baseSize * 1.5;
        const clickRadius = starOuterRadius; // Use outer radius for click detection

        // Draw the base image
        ctx.drawImage(data.image, 0, 0, canvas.width, canvas.height);

        // Draw selected masks
        const maskControlsDiv = canvas.nextElementSibling.querySelector('.mask-controls'); // Find controls next to canvas
        const visibleMaskIds = new Set();
        if (maskControlsDiv) {
            const checkboxes = maskControlsDiv.querySelectorAll('input[type="checkbox"]:checked');
            checkboxes.forEach(cb => visibleMaskIds.add(cb.value));
        }

        data.masks.forEach(mask => {
            if (visibleMaskIds.has(mask.instance_id)) {
                const maskImg = maskImageCache[mask.mask_data_uri];
                if (maskImg && maskImg.complete) {
                    ctx.globalAlpha = 0.5; // Apply transparency
                    ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
                    ctx.globalAlpha = 1.0; // Reset alpha
                } else if (!maskImg) {
                    console.warn(`Mask image not preloaded or failed for ${mask.instance_id}`);
                }
            }
        });

        // --- Draw Points Based on Mode ---
        const modeSelector = `input[name="point-mode-${canvasId}"]:checked`;
        const selectedModeInput = document.querySelector(modeSelector);
        const mode = selectedModeInput ? selectedModeInput.value : 'used'; // Default to 'used' if not found

        let pointsToDraw = [];
        if (mode === 'all') {
            pointsToDraw = data.prior_points;
        } else { // Default to 'used'
            pointsToDraw = data.used_points;
        }

        data.clickablePoints = []; // Clear previous clickable points
        const pointColorAllMode = 'cyan'; // New brighter color for 'all' mode

        pointsToDraw.forEach(point => {
            const x = point.x;
            const y = point.y;
            const label = point.label;

            if (label === 0) {
                // Always draw Background points as RED Squares
                ctx.fillStyle = 'red'; // Always red for background points
                ctx.beginPath();
                ctx.rect(x - squareSize / 2, y - squareSize / 2, squareSize, squareSize);
                ctx.fill();
                // Optional: Add outline
                // ctx.strokeStyle = 'black';
                // ctx.lineWidth = 0.5;
                // ctx.stroke();
            } else {
                // Foreground points (label > 0)
                if (mode === 'used') {
                    // Draw Stars for Used Foreground Points
                    ctx.fillStyle = 'lime';
                    ctx.strokeStyle = 'black';
                    ctx.lineWidth = 0.5;
                    drawStar(ctx, x, y, starOuterRadius, starInnerRadius, 5); // Use dynamic sizes
                    ctx.fill();
                    // ctx.stroke();
                } else {
                    // Draw Dots for All Foreground Prior Points
                    ctx.fillStyle = pointColorAllMode; // Use new color
                    ctx.beginPath();
                    ctx.arc(x, y, dotRadius, 0, 2 * Math.PI); // Use dynamic size
                    ctx.fill();
                }
            }

            // Store point info for potential clicks (regardless of mode)
            data.clickablePoints.push({
                x: x,
                y: y,
                radius: clickRadius, // Use consistent click radius based on star size
                info: point // Store the whole point object
            });
        });
        // --- End Point Drawing ---
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
