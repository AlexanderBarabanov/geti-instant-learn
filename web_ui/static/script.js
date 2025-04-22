document.addEventListener("DOMContentLoaded", () => {
  const processButton = document.getElementById("runButton");
  const pipelineSelect = document.getElementById("pipelineSelect");
  const samNameSelect = document.getElementById("samNameSelect");
  const datasetSelect = document.getElementById("datasetSelect");
  const classNameSelect = document.getElementById("classNameSelect");
  const nShotInput = document.getElementById("nShotInput");
  const numBackgroundPointsInput = document.getElementById(
    "numBackgroundPointsInput",
  );
  const resultsContainer = document.getElementById("results-container");
  // Progress bar elements
  const progressContainer = document.getElementById("progress-container");
  const progressBarFill = document.getElementById("progress-bar-fill");
  const progressText = document.getElementById("progress-text"); // Optional text

  const MAX_CANVAS_WIDTH = 500;

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

  async function fetchAndPopulateClasses (selectedDataset) {
    classNameSelect.innerHTML =
      '<option value="" disabled selected>Loading...</option>';
    try {
      const response = await fetch(
        `/api/classes?dataset=${encodeURIComponent(selectedDataset)}`,
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.error || `HTTP error! status: ${response.status}`,
        );
      }
      const data = await response.json();
      classNameSelect.innerHTML =
        '<option value="" disabled selected>Select Class...</option>';
      if (data.classes && data.classes.length > 0) {
        classNameSelect.innerHTML =
          '<option value="" disabled>Select Class...</option>';
        data.classes.forEach((className) => {
          const option = document.createElement("option");
          option.value = className;
          option.textContent = className;
          classNameSelect.appendChild(option);
        });
        // Pre-select the first actual class (index 1, since index 0 is the disabled placeholder)
        if (classNameSelect.options.length > 1) {
          classNameSelect.selectedIndex = 1;
        }
      } else {
        classNameSelect.innerHTML =
          '<option value="" disabled selected>No classes found</option>';
      }
    } catch (error) {
      console.error("Error fetching classes:", error);
      classNameSelect.innerHTML =
        '<option value="" disabled selected>Error loading</option>';
    }
  }

  // Event listener for dataset change
  datasetSelect.addEventListener("change", () => {
    console.log("Dataset select changed!");
    fetchAndPopulateClasses(datasetSelect.value);
  });
  fetchAndPopulateClasses(datasetSelect.value);

  processButton.addEventListener("click", async () => {
    const pipeline = pipelineSelect.value;
    const samName = samNameSelect.value;
    const dataset = datasetSelect.value;
    const className = classNameSelect.value;
    const nShot = parseInt(nShotInput.value, 10);
    const numBackgroundPoints = parseInt(numBackgroundPointsInput.value, 10);

    if (!className || isNaN(nShot) || nShot < 1) {
      // Use progress text for errors before starting
      progressContainer.classList.remove("hidden");
      progressText.textContent = "Invalid input.";
      progressBarFill.style.width = "0%";
      progressBarFill.classList.add("bg-red-600"); // Make bar red on error
      progressBarFill.classList.remove("bg-blue-600");
      return;
    }

    // --- UI Update for Loading State --- 
    const originalButtonText = processButton.innerHTML;
    processButton.disabled = true;
    processButton.innerHTML = `
            <svg class="animate-spin -ml-1 mr-2 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Processing...
        `;
    resultsContainer.innerHTML = "";
    // Show and reset progress bar
    progressContainer.classList.remove("hidden");
    progressBarFill.style.width = "0%";
    progressBarFill.classList.remove("bg-red-600"); // Ensure it's blue
    progressBarFill.classList.add("bg-blue-600");
    progressText.textContent = "Initializing..."; // Initial text
    Object.keys(canvasDataStore).forEach((key) => delete canvasDataStore[key]);
    Object.keys(maskImageCache).forEach((key) => delete maskImageCache[key]);

    // --- Fetch and Process Streamed Response --- 
    let totalTargets = 0;
    let resultsCount = 0;
    let errorOccurred = false; // Flag to track errors

    try {
      const response = await fetch("/api/process", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          pipeline: pipeline,
          dataset: dataset,
          class_name: className,
          n_shot: nShot,
          num_background_points: numBackgroundPoints,
          sam_name: samName,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` }));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      // --- Stream Reading Logic --- 
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let isFirstChunk = true; // Flag to handle the first message
      console.log("Starting stream reading loop...");

      while (true) {
        console.log("Waiting for reader.read()...");
        const { done, value } = await reader.read();
        console.log(`Read chunk: done=${done}, value size=${value?.length}`);

        if (done) {
          console.log("Stream finished.");
          break;
        }

        const decodedChunk = decoder.decode(value, { stream: true });
        console.log("Decoded chunk:", decodedChunk);
        buffer += decodedChunk;
        console.log("Current buffer:", buffer);

        let newlineIndex;
        while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
          const line = buffer.substring(0, newlineIndex).trim();
          buffer = buffer.substring(newlineIndex + 1);
          console.log("Processing line:", line);

          if (line) {
            try {
              const dataChunk = JSON.parse(line);
              console.log("Parsed data chunk:", dataChunk);

              if (isFirstChunk && dataChunk.total_targets !== undefined) {
                // --- Handle Total Count --- 
                totalTargets = parseInt(dataChunk.total_targets, 10) || 0;
                console.log("Received total targets:", totalTargets);
                progressText.textContent = `Processing 0 / ${totalTargets}...`;
                isFirstChunk = false; // Move to processing results
              } else if (dataChunk.error) {
                // --- Handle Backend Error Chunk ---
                console.error("Backend processing error:", dataChunk.error);
                progressText.textContent = `Error: ${dataChunk.error}`;
                progressBarFill.classList.remove("bg-blue-600");
                progressBarFill.classList.add("bg-red-600");
                errorOccurred = true;
              } else if (dataChunk.target_results) {
                // --- Handle Results Chunk --- 
                if (errorOccurred) continue;

                const receivedCount = dataChunk.target_results.length;
                resultsCount += receivedCount;
                console.log(`Calling displayResults for ${receivedCount} items.`);
                displayResults(dataChunk.target_results); // Append results incrementally

                // Update progress bar
                if (totalTargets > 0) {
                  const percentage = Math.min(100, (resultsCount / totalTargets) * 100);
                  progressBarFill.style.width = `${percentage}%`;
                  progressText.textContent = `Processing ${resultsCount} / ${totalTargets}...`;
                } else {
                  progressText.textContent = `Received ${resultsCount} results... (Total unknown)`;
                }
              }
            } catch (e) {
              console.error("Error parsing JSON chunk:", e, "Chunk:", line);
              progressText.textContent = "Error reading results stream.";
              progressBarFill.classList.remove("bg-blue-600");
              progressBarFill.classList.add("bg-red-600");
              errorOccurred = true;
              break; // Exit inner loop on parse error
            }
          }
        }
        if (errorOccurred) break; // Exit outer loop if parse error occurred
      }
      // --- End Stream Reading Logic --- 

      if (!errorOccurred) {
        if (resultsCount === 0 && totalTargets === 0 && !isFirstChunk) { // Check if total was received but no results
          progressText.textContent = "Processing complete. No results returned.";
        } else if (resultsCount === totalTargets) {
          progressBarFill.style.width = `100%`;
          progressText.textContent = `Processing complete (${resultsCount} / ${totalTargets}).`;
        } else if (resultsCount < totalTargets) {
          progressText.textContent = `Processing incomplete (${resultsCount} / ${totalTargets}). Stream ended unexpectedly.`;
          progressBarFill.classList.remove("bg-blue-600");
          progressBarFill.classList.add("bg-yellow-500"); // Indicate warning
        }
      }

    } catch (error) {
      console.error("Error fetching or processing data:", error);
      // Show error in progress text area
      progressContainer.classList.remove("hidden");
      progressText.textContent = `Error: ${error.message}`;
      progressBarFill.style.width = "100%"; // Fill bar on error
      progressBarFill.classList.remove("bg-blue-600");
      progressBarFill.classList.add("bg-red-600");
      errorOccurred = true;
    } finally {
      console.log("Executing finally block...");
      processButton.disabled = false;
      console.log(`Run button disabled state: ${processButton.disabled}`);
      processButton.innerHTML = originalButtonText;
    }
  });

  function displayResults (targetResults) {
    targetResults.forEach((result, indexOffset) => {
      // Calculate a unique index based on how many results are already displayed
      const existingResultsCount = resultsContainer.children.length;
      const uniqueIndex = existingResultsCount + indexOffset; // Simple way to get unique index

      const targetItemDiv = document.createElement("div");
      targetItemDiv.classList.add(
        "target-item",
        "bg-white",
        "p-4",
        "rounded-lg",
        "shadow",
        "flex",
        "flex-col",
        "gap-4",
      );
      targetItemDiv.id = `target-item-${uniqueIndex}`; // Use unique ID

      const canvasContainer = document.createElement("div");
      canvasContainer.classList.add("flex-shrink-0");

      const canvasId = `canvas-${uniqueIndex}`; // Use unique ID for canvas
      const canvas = document.createElement("canvas");
      canvas.id = canvasId;
      canvas.classList.add(
        "border",
        "border-gray-300",
        "rounded",
        "max-w-full",
      );
      canvasContainer.appendChild(canvas);

      const controlsContainer = document.createElement("div");
      controlsContainer.classList.add(
        "item-controls",
        "flex-grow",
        "space-y-4",
      );

      // --- Point Display Mode Grouped Buttons ---
      const pointModeHeader = document.createElement("strong");
      pointModeHeader.classList.add(
        "block",
        "text-sm",
        "font-medium",
        "text-gray-700",
        "mb-1",
      );
      pointModeHeader.textContent = "Point Display Mode:";
      controlsContainer.appendChild(pointModeHeader);

      const pointModeGroup = document.createElement("div");
      pointModeGroup.classList.add("inline-flex", "rounded-md", "shadow-sm");
      pointModeGroup.setAttribute("role", "group");
      const pointModeGroupId = `point-mode-group-${canvasId}`;
      pointModeGroup.id = pointModeGroupId;

      // Base classes for buttons
      const buttonBaseClasses = [
        "relative",
        "inline-flex",
        "items-center",
        "px-3",
        "py-2",
        "text-sm",
        "font-semibold",
        "ring-1",
        "ring-inset",
        "ring-gray-300",
        "focus:z-10",
      ];
      const buttonSelectedClasses = [
        "bg-primary",
        "text-white",
        "hover:bg-blue-700",
      ];
      const buttonUnselectedClasses = [
        "bg-white",
        "text-gray-900",
        "hover:bg-gray-50",
      ];

      // Used Points Button
      const buttonUsed = document.createElement("button");
      buttonUsed.type = "button";
      buttonUsed.textContent = "Used Points";
      buttonUsed.dataset.pointMode = "used";
      buttonUsed.classList.add(
        ...buttonBaseClasses,
        "rounded-l-md",
        ...buttonSelectedClasses,
      );

      // All Points Button
      const buttonAll = document.createElement("button");
      buttonAll.type = "button";
      buttonAll.textContent = "All Points";
      buttonAll.dataset.pointMode = "all";
      buttonAll.classList.add(
        ...buttonBaseClasses,
        "-ml-px",
        "rounded-r-md",
        ...buttonUnselectedClasses,
      );

      pointModeGroup.appendChild(buttonUsed);
      pointModeGroup.appendChild(buttonAll);
      controlsContainer.appendChild(pointModeGroup);

      // Event listener for the group
      pointModeGroup.addEventListener("click", (event) => {
        console.log("Point mode group clicked!");
        const clickedButton = event.target.closest("button");
        if (!clickedButton || !pointModeGroup.contains(clickedButton)) return;

        const newMode = clickedButton.dataset.pointMode;
        const currentMode = canvasDataStore[canvasId].pointMode;

        if (newMode !== currentMode) {
          canvasDataStore[canvasId].pointMode = newMode;

          // Update styles
          const buttons = pointModeGroup.querySelectorAll("button");
          buttons.forEach((button) => {
            button.classList.remove(
              ...buttonSelectedClasses,
              ...buttonUnselectedClasses,
            );
            if (button.dataset.pointMode === newMode) {
              button.classList.add(...buttonSelectedClasses);
            } else {
              button.classList.add(...buttonUnselectedClasses);
            }
          });

          redrawCanvas(canvasId);
        }
      });

      const maskControlsDiv = document.createElement("div");
      maskControlsDiv.classList.add("mask-controls", "space-y-2");
      maskControlsDiv.innerHTML =
        '<strong class="block text-sm font-medium text-gray-700 mb-1">Masks:</strong>';

      const maskButtonContainer = document.createElement("div");
      maskButtonContainer.classList.add("flex", "gap-x-2", "mb-2");

      // Function to create styled buttons
      const createButton = (text) => {
        const button = document.createElement("button");
        button.textContent = text;
        button.classList.add(
          "px-2.5",
          "py-1.5",
          "border",
          "border-gray-300",
          "rounded-md",
          "shadow-sm",
          "text-xs",
          "font-medium",
          "text-gray-700",
          "bg-white",
          "hover:bg-gray-50",
          "focus:outline-none",
          "focus:ring-2",
          "focus:ring-offset-2",
          "focus:ring-indigo-500",
        );
        button.dataset.canvasId = canvasId;
        return button;
      };

      const selectAllButton = createButton("Select All");
      selectAllButton.addEventListener("click", (event) => {
        const targetCanvasId = event.target.dataset.canvasId;
        const maskCheckboxesContainer = document.getElementById(
          `mask-checkboxes-${targetCanvasId}`,
        );
        if (maskCheckboxesContainer) {
          const checkboxes = maskCheckboxesContainer.querySelectorAll(
            'input[type="checkbox"]',
          );
          checkboxes.forEach((cb) => (cb.checked = true));
          redrawCanvas(targetCanvasId);
        }
      });

      const unselectAllButton = createButton("Unselect All");
      unselectAllButton.addEventListener("click", (event) => {
        const targetCanvasId = event.target.dataset.canvasId;
        const maskCheckboxesContainer = document.getElementById(
          `mask-checkboxes-${targetCanvasId}`,
        );
        if (maskCheckboxesContainer) {
          const checkboxes = maskCheckboxesContainer.querySelectorAll(
            'input[type="checkbox"]',
          );
          checkboxes.forEach((cb) => (cb.checked = false));
          redrawCanvas(targetCanvasId);
        }
      });

      maskButtonContainer.appendChild(selectAllButton);
      maskButtonContainer.appendChild(unselectAllButton);
      maskControlsDiv.appendChild(maskButtonContainer);

      // Container for the actual checkboxes
      const maskCheckboxesContainer = document.createElement("div");
      maskCheckboxesContainer.id = `mask-checkboxes-${canvasId}`;
      maskCheckboxesContainer.classList.add(
        "space-y-1",
        "overflow-y-auto",
        "max-h-40",
      ); // Spacing between checkboxes
      maskControlsDiv.appendChild(maskCheckboxesContainer);

      controlsContainer.appendChild(maskControlsDiv);

      targetItemDiv.appendChild(canvasContainer);
      targetItemDiv.appendChild(controlsContainer);
      resultsContainer.appendChild(targetItemDiv);

      // Store data needed for redraws - use unique canvasId
      canvasDataStore[canvasId] = {
        image: null,
        masks: result.masks || [],
        points: {
          used: result.used_points || [],
          all: (result.prior_points || []).concat(result.used_points || [])
        },
        element: canvas,
        clickablePoints: [],
        pointMode: "used",
        scaleX: 1,
        scaleY: 1,
        originalWidth: 0,
        originalHeight: 0,
        similarityMaps: result.similarity_maps || [] // Store available maps
      };

      // --- Similarity Map Display (Applying Tailwind) ---
      console.log(
        `[Canvas ${canvasId}] Checking for similarity maps:`,
        result.similarity_maps,
      );
      const simMaps = canvasDataStore[canvasId].similarityMaps;
      if (simMaps && simMaps.length > 0) {
        console.log(
          `[Canvas ${canvasId}] Found ${simMaps.length} similarity map(s). Creating container.`,
        );
        const simMapContainer = document.createElement("div");
        simMapContainer.classList.add("similarity-map-container");
        simMapContainer.innerHTML =
          '<h5 class="text-lg font-medium text-gray-900 mb-2">Similarity Maps:</h5>';

        const simMapGrid = document.createElement("div");
        simMapGrid.classList.add(
          "grid",
          "grid-cols-2",
          "sm:grid-cols-3",
          "md:grid-cols-4",
          "lg:grid-cols-5",
          "gap-2",
        ); // Adjusted grid cols and gap

        simMaps.forEach((mapData, mapIndex) => {
          const mapDiv = document.createElement("div");

          const mapImg = document.createElement("img");
          mapImg.src = mapData.map_data_uri;
          mapImg.alt = `Similarity Map ${mapIndex + 1} (Point ${mapData.point_index})`;
          mapImg.classList.add(
            "w-full",
            "h-auto",
            "border",
            "border-gray-300",
            "rounded",
          ); // Fill width, auto height
          mapImg.dataset.canvasId = canvasId;
          mapImg.dataset.mapUri = mapData.map_data_uri;

          const uriLength = mapData.map_data_uri
            ? mapData.map_data_uri.length
            : 0;
          console.log(
            `[Canvas ${canvasId}] Setting sim map ${mapIndex} src (length: ${uriLength}): ${mapData.map_data_uri ? mapData.map_data_uri.substring(0, 80) + "..." : "null/empty"}`,
          );

          mapImg.onerror = () => {
            // Add error handler for sim map images
            console.error(
              `[Canvas ${canvasId}] Failed to load similarity map image ${mapIndex} from data URI (length: ${uriLength}).`,
            );
            // Optionally show an error placeholder, or just leave the div empty/hide it
            mapDiv.innerHTML = `<div class="w-full aspect-square bg-gray-100 flex items-center justify-center border border-red-300 rounded text-red-600 text-xs p-1">Error loading map</div>`;
          };

          mapDiv.appendChild(mapImg); // Append image directly to mapDiv
          simMapGrid.appendChild(mapDiv);
        });

        simMapContainer.appendChild(simMapGrid);
        // Make sure controlsContainer exists before appending
        const controlsContainer = targetItemDiv.querySelector('.item-controls');
        if (controlsContainer) {
          controlsContainer.appendChild(simMapContainer);
        } else {
          console.error("Could not find controls container for sim maps", targetItemDiv);
        }
      }

      canvas.addEventListener("click", handleCanvasClick);

      // Create checkboxes and add listeners
      if (result.masks && result.masks.length > 0) {
        result.masks.forEach((mask, maskIndex) => {
          const checkboxId = `${canvasId}-mask-${mask.instance_id}`;

          // Create container for checkbox + label for styling
          const maskDiv = document.createElement("div");
          maskDiv.classList.add("flex", "items-center");

          const label = document.createElement("label");
          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.id = checkboxId;
          checkbox.value = mask.instance_id;
          checkbox.checked = true; // Default to visible
          checkbox.dataset.canvasId = canvasId; // Link checkbox to canvas
          // Apply Tailwind classes to checkbox
          checkbox.classList.add(
            "h-4",
            "w-4",
            "text-indigo-600",
            "border-gray-300",
            "rounded",
            "focus:ring-indigo-500",
          );

          label.htmlFor = checkboxId;
          label.textContent = `Mask ${maskIndex + 1}`; // Keep text
          label.classList.add("ml-2", "block", "text-sm"); // Basic label styling

          maskDiv.appendChild(checkbox);
          maskDiv.appendChild(label);
          maskCheckboxesContainer.appendChild(maskDiv); // Append the container div

          // Add listener to redraw when checkbox changes
          checkbox.addEventListener("change", (event) => {
            const targetCanvasId = event.target.dataset.canvasId;
            redrawCanvas(targetCanvasId);
          });
        });
      } else {
        maskCheckboxesContainer.innerHTML += "<i>No masks found</i>";
      }

      // Preload mask images using data URIs
      preloadMaskImages(canvasId);

      // Load base image onto canvas using data URI
      const img = new Image();
      img.onload = () => {
        // Store original dimensions
        const naturalWidth = img.naturalWidth;
        const naturalHeight = img.naturalHeight;
        canvasDataStore[canvasId].originalWidth = naturalWidth;
        canvasDataStore[canvasId].originalHeight = naturalHeight;

        // Calculate scaled dimensions
        let targetWidth = naturalWidth;
        let targetHeight = naturalHeight;

        if (naturalWidth > MAX_CANVAS_WIDTH) {
          const scaleRatio = MAX_CANVAS_WIDTH / naturalWidth;
          targetWidth = MAX_CANVAS_WIDTH;
          targetHeight = naturalHeight * scaleRatio;
        }

        canvas.width = targetWidth;
        canvas.height = targetHeight;

        // Store scaling factors
        canvasDataStore[canvasId].scaleX = targetWidth / naturalWidth;
        canvasDataStore[canvasId].scaleY = targetHeight / naturalHeight;

        canvasDataStore[canvasId].image = img;
        redrawCanvas(canvasId);
      };
      img.onerror = () => {
        console.error(`Failed to load image from data URI`);
        const ctx = canvas.getContext("2d");
        canvas.width = 300;
        canvas.height = 100;
        ctx.fillStyle = "red";
        ctx.font = "16px sans-serif";
        ctx.fillText("Error loading image", 10, 50);
      };
      img.src = result.image_data_uri; // Use data URI directly
    });
  }

  function preloadMaskImages (canvasId) {
    const data = canvasDataStore[canvasId];
    if (!data || !data.masks) return;

    data.masks.forEach((mask) => {
      const uri = mask.mask_data_uri;
      if (uri && !maskImageCache[uri]) {
        // Check if URI exists
        const maskImg = new Image();
        maskImg.onload = () => {
          maskImageCache[uri] = maskImg; // Store loaded image in cache
        };
        maskImg.onerror = () => {
          console.error(`Failed to preload mask image from data URI`);
          maskImageCache[uri] = null; // Mark as failed
        };
        maskImageCache[uri] = "loading"; // Mark as loading
        maskImg.src = uri; // Use data URI directly
      }
    });
  }

  function drawStar (ctx, x, y, outerRadius, innerRadius, points) {
    ctx.beginPath();
    ctx.moveTo(x, y - outerRadius);
    for (let i = 0; i < points; i++) {
      let angle = (Math.PI / points) * (2 * i + 1.5);
      ctx.lineTo(
        x + outerRadius * Math.cos(angle),
        y + outerRadius * Math.sin(angle),
      );
      angle = (Math.PI / points) * (2 * i + 2.5);
      ctx.lineTo(
        x + innerRadius * Math.cos(angle),
        y + innerRadius * Math.sin(angle),
      );
    }
    ctx.closePath();
  }

  function redrawCanvas (canvasId) {
    const data = canvasDataStore[canvasId];
    if (!data || !data.image) {
      return;
    }

    const mode = data.pointMode || "used";
    const scaleX = data.scaleX || 1;
    const scaleY = data.scaleY || 1;
    const canvas = data.element;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const baseSize = Math.max(3, Math.min(canvas.width, canvas.height) * 0.01);
    const squareSize = baseSize * 1.5;
    ctx.drawImage(data.image, 0, 0, canvas.width, canvas.height);

    // Draw selected masks
    const maskCheckboxesContainer = document.getElementById(
      `mask-checkboxes-${canvasId}`,
    );
    const visibleMaskIds = new Set();
    if (maskCheckboxesContainer) {
      const checkboxes = maskCheckboxesContainer.querySelectorAll(
        'input[type="checkbox"]:checked',
      );
      checkboxes.forEach((cb) => visibleMaskIds.add(cb.value));
    } else {
      console.warn(
        `Could not find mask checkboxes container for canvas: ${canvasId}`,
      );
    }

    data.masks.forEach((mask) => {
      if (visibleMaskIds.has(mask.instance_id)) {
        const maskImg = maskImageCache[mask.mask_data_uri];
        if (maskImg && maskImg.complete) {
          ctx.globalAlpha = 0.5;
          ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
          ctx.globalAlpha = 1.0;
        } else if (!maskImg) {
          console.warn(
            `Mask image not preloaded or failed for ${mask.instance_id}`,
          );
        }
      }
    });

    let pointsToDraw = [];
    if (mode === "all") {
      pointsToDraw = data.points.all || [];
    } else {
      pointsToDraw = data.points.used || [];
    }

    data.clickablePoints = [];

    const OUTLINE_COLOR = "rgba(255, 255, 255, 0.8)";
    const OUTLINE_WIDTH = 1.5;
    const FOREGROUND_USED_COLOR = "rgba(50, 205, 50, 1)";
    const FOREGROUND_ALL_COLOR = "rgba(135, 206, 250, 1)";
    const BACKGROUND_COLOR = "rgba(255, 0, 0, 1)";

    const pointRadius = baseSize * 1.2;
    const clickRadius = pointRadius * 1.5;

    pointsToDraw.forEach((point) => {
      const drawX = point.x * scaleX;
      const drawY = point.y * scaleY;
      const label = point.label;

      if (label === 0) {
        // Draw Background points as solid RED squares with white outline
        ctx.fillStyle = BACKGROUND_COLOR;
        ctx.strokeStyle = OUTLINE_COLOR;
        ctx.lineWidth = OUTLINE_WIDTH;
        ctx.beginPath();
        ctx.rect(
          drawX - squareSize / 2,
          drawY - squareSize / 2,
          squareSize,
          squareSize,
        );
        ctx.fill();
        ctx.stroke();
      } else {
        // Foreground points (label > 0)
        if (mode === "used") {
          // Draw Used Foreground points as solid GREEN circles with white outline
          ctx.fillStyle = FOREGROUND_USED_COLOR;
          ctx.strokeStyle = OUTLINE_COLOR;
          ctx.lineWidth = OUTLINE_WIDTH;
          ctx.beginPath();
          ctx.arc(drawX, drawY, pointRadius, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();
        } else {
          // Draw All Foreground points as filled Light Blue circles with white outline
          ctx.fillStyle = FOREGROUND_ALL_COLOR;
          ctx.strokeStyle = OUTLINE_COLOR;
          ctx.lineWidth = OUTLINE_WIDTH;
          ctx.beginPath();
          ctx.arc(drawX, drawY, pointRadius, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();
        }
      }

      // Store point info for potential clicks using ORIGINAL coordinates
      data.clickablePoints.push({
        x: point.x,
        y: point.y,
        radius: clickRadius,
        info: point,
      });
    });
  }

  function handleCanvasClick (event) {
    const canvas = event.target;
    const canvasId = canvas.id;
    const data = canvasDataStore[canvasId];
    if (!data) return;

    const rect = canvas.getBoundingClientRect();
    const clickScaleX = canvas.width / rect.width;
    const clickScaleY = canvas.height / rect.height;
    const clickX = (event.clientX - rect.left) * clickScaleX;
    const clickY = (event.clientY - rect.top) * clickScaleY;

    // Retrieve canvas scaling factors
    // These are the factors needed to scale original point coords to displayed size
    const canvasScaleX = data.scaleX || 1;
    const canvasScaleY = data.scaleY || 1;

    let clickedOnPoint = false;

    // Check if the click hit any stored foreground point areas
    // Iterate in reverse to prioritize points drawn on top if overlap occurs
    for (let i = data.clickablePoints.length - 1; i >= 0; i--) {
      const pt = data.clickablePoints[i];

      const scaledPtX = pt.x * canvasScaleX;
      const scaledPtY = pt.y * canvasScaleY;
      const distance = Math.sqrt(
        Math.pow(clickX - scaledPtX, 2) + Math.pow(clickY - scaledPtY, 2),
      );

      // Check if click is near a foreground point (label > 0)
      if (distance <= pt.radius && pt.info.label > 0) {
        clickedOnPoint = true;

        // --- Find the mask associated with the clicked point ---
        const currentMode = data.pointMode || "used";
        const pointsList =
          currentMode === "all" ? data.points.all : data.points.used;

        let pointIndex = -1;
        pointIndex = pointsList.findIndex((p) => p === pt.info);

        // Only foreground points (label > 0) should trigger mask toggling
        // And only if we are in 'used' mode and found a valid index
        if (
          pt.info.label > 0 &&
          pointIndex !== -1 &&
          data.masks &&
          pointIndex < data.masks.length &&
          currentMode === "used"
        ) {
          const targetInstanceId = data.masks[pointIndex].instance_id;
          const checkboxId = `${canvasId}-mask-${targetInstanceId}`;
          const checkbox = document.getElementById(checkboxId);
          if (checkbox) {
            checkbox.checked = !checkbox.checked;
            redrawCanvas(canvasId);
          }
        } else {
          console.warn(
            `No mask found for clicked point's index: ${pointIndex}`,
          );
        }
        break; // Stop checking other points once one is found
      }
    }
  }
});
