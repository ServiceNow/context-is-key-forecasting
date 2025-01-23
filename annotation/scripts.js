// Show the definitions modal
function showDefinitions() {
    const modal = document.getElementById("definitions-modal");
    modal.style.display = "flex"; // Show the modal
}

// Close the definitions modal
function closeDefinitions() {
    const modal = document.getElementById("definitions-modal");
    modal.style.display = "none"; // Hide the modal
}

// Update the progress bar
function updateProgressBar() {
    const dictionary = JSON.parse(localStorage.getItem("cik_capability_annotations")) || {};
    const totalTasks = Object.keys(dictionary).length;
    const annotatedTasks = Object.values(dictionary).filter(task => task.annotations.length > 0).length;

    const progressPercentage = totalTasks > 0 ? (annotatedTasks / totalTasks) * 100 : 0;
    document.getElementById("progress-bar").style.width = `${progressPercentage}%`;

    // Update progress info
    const progressInfo = `${annotatedTasks}/${totalTasks}`;
    document.getElementById("progress-info").textContent = progressInfo;
}

// Fetch and render a task page
async function fetchAndRenderPage(taskUrl) {
    try {
        const response = await fetch(taskUrl);
        if (!response.ok) {
            throw new Error(`Failed to fetch page: ${response.statusText}`);
        }

        const htmlText = await response.text();

        // Parse the HTML content
        const parser = new DOMParser();
        const doc = parser.parseFromString(htmlText, "text/html");

        // Remove <p> elements containing "Capabilities:" or "Types of context:"
        const paragraphs = doc.querySelectorAll("p");
        paragraphs.forEach(p => {
            if (p.textContent.includes("Capabilities:") || p.textContent.includes("Types of context:")) {
                p.remove();
            }
        });

        // Remove all buttons with class "back-button"
        const backButtons = doc.querySelectorAll("button.back-button");
        backButtons.forEach(button => button.remove());

        // Copy stylesheets and inline styles
        const stylesheets = Array.from(doc.querySelectorAll('link[rel="stylesheet"], style'));
        const renderedContent = document.getElementById("rendered-content");
        renderedContent.innerHTML = ""; // Clear previous content

        stylesheets.forEach(style => {
            const clonedStyle = style.cloneNode(true);
            document.head.appendChild(clonedStyle);
        });

        // Render the modified content in the #rendered-content div
        renderedContent.style.display = "block";
        renderedContent.innerHTML = doc.body.innerHTML;

        // Show the annotation pane
        document.getElementById("annotation-pane").style.display = "block";

    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// Save annotations and move to the next task
function saveAnnotationsAndNext() {
    const dictionary = JSON.parse(localStorage.getItem("cik_capability_annotations")) || {};
    const currentTask = Object.keys(dictionary).find(task => dictionary[task].annotations.length === 0);

    if (!currentTask) {
        alert("All tasks are annotated!");
        return;
    }

    // Get the form
    const form = document.getElementById("capabilities-form");

    // Get selected capabilities (checkboxes)
    const selectedCapabilities = Array.from(form.elements)
        .filter(input => input.checked && input.type === "checkbox")
        .map(input => input.value);

    // Get selected yes/no radio
    const selectedRadio = Array.from(form.elements)
        .find(input => input.checked && input.type === "radio")?.value;

    // Save both sets of data into the dictionary
    dictionary[currentTask].annotations = selectedCapabilities;
    dictionary[currentTask].yesNoResponse = selectedRadio;

    // Update localStorage
    localStorage.setItem("cik_capability_annotations", JSON.stringify(dictionary));

    // Reset form and re-check button state
    form.reset();
    toggleNextButtonState();

    // Update progress bar and move on
    updateProgressBar();
    openRandomUnannotatedTask();
}

// Skip the current task and load a random one
function skipTask() {
    document.getElementById("capabilities-form").reset();
    openRandomUnannotatedTask();
}

// Open a random unannotated task
function openRandomUnannotatedTask() {
    const dictionary = JSON.parse(localStorage.getItem("cik_capability_annotations")) || {};
    const unannotatedTasks = Object.values(dictionary).filter(task => task.annotations.length === 0);

    if (unannotatedTasks.length > 0) {
        // Select a random unannotated task
        const randomIndex = Math.floor(Math.random() * unannotatedTasks.length);
        const randomTask = unannotatedTasks[randomIndex];

        // Load the random task
        fetchAndRenderPage(randomTask.url);
    } else {
        alert("All tasks are annotated!");
    }
}

// Helper function to shuffle an array
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

// Fetch tasks from the source URL
async function fetchTasks() {
    const pageUrl = document.getElementById("page-url").value;
    const existingData = localStorage.getItem("cik_capability_annotations");

    if (existingData) {
        const overwrite = confirm("Found existing annotations in memory. Do you want to overwrite them?");
        if (!overwrite) {
            updateProgressBar();
            return openRandomUnannotatedTask();
        }
    }

    try {
        // Fetch the HTML content of the page
        const response = await fetch(pageUrl);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const text = await response.text();

        // Parse the HTML content
        const parser = new DOMParser();
        const doc = parser.parseFromString(text, "text/html");

        // Find all <a> elements within <li>
        const anchors = doc.querySelectorAll("li a");

        // Create an array of task entries
        const tasks = [];
        anchors.forEach(anchor => {
            const href = anchor.getAttribute("href")?.trim(); // Get the href attribute
            const text = anchor.textContent.trim(); // Get the text content of the list item
            if (href && text) {
                const fullUrl = new URL(href, pageUrl).href; // Resolve relative URL
                tasks.push([text, { url: fullUrl, annotations: [] }]); // Add as a key-value pair
            }
        });

        // Shuffle the tasks
        const randomizedTasks = shuffleArray(tasks);

        // Convert the shuffled array back to a dictionary
        const dictionary = Object.fromEntries(randomizedTasks);

        // Store the dictionary in session storage
        localStorage.setItem("cik_capability_annotations", JSON.stringify(dictionary));

        // Update progress and open the first unannotated task
        updateProgressBar();
        openRandomUnannotatedTask();
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// Download annotations as a JSON file
function downloadJSON() {
    const dictionary = JSON.parse(localStorage.getItem("cik_capability_annotations"));
    if (!dictionary) {
        alert("No data to download!");
        return;
    }

    const totalTasks = Object.keys(dictionary).length;
    const annotatedTasks = Object.values(dictionary).filter(task => task.annotations.length > 0).length;

    // Check if all tasks are annotated
    if (annotatedTasks < totalTasks) {
        const proceed = confirm(`Only ${annotatedTasks}/${totalTasks} tasks are annotated. Are you sure you want to download partial results?`);
        if (!proceed) return; // Exit if the user cancels
    }

    const blob = new Blob([JSON.stringify(dictionary, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cik_capability_annotations.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Disable the Next button initially
function toggleNextButtonState() {
    const form = document.getElementById("capabilities-form");
    const checkboxes = Array.from(form.elements).filter(input => input.type === "checkbox");
    const yesNoButtons = Array.from(form.elements).filter(input => input.type === "radio");
    const nextButton = form.querySelector(".next-button");

    // Enable the Next button if at least one checkbox is checked
    const isAnyChecked = checkboxes.some(checkbox => checkbox.checked);
    const isYesNoSelected = yesNoButtons.some(button => button.checked);
    nextButton.disabled = !isAnyChecked || !isYesNoSelected; // Disable if no checkboxes are checked and no radio buttons are selected
}

// Attach event listeners to checkboxes to update the button state
function initializeCheckboxListeners() {
    const form = document.getElementById("capabilities-form");
    const checkboxes = Array.from(form.elements).filter(input => input.type === "checkbox");
    const yesNoButtons = Array.from(form.elements).filter(input => input.type === "radio");

    checkboxes.forEach(checkbox => {
        checkbox.addEventListener("change", toggleNextButtonState);
    });
    yesNoButtons.forEach(button => {
        button.addEventListener("change", toggleNextButtonState);
    });

    // Ensure the button state is updated on page load
    toggleNextButtonState();
}

// Call initializeCheckboxListeners on page load
window.onload = function () {
    const defaultUrl = new URLSearchParams(window.location.search).get("url") || "https://servicenow.github.io/context-is-key-forecasting/v0/";
    document.getElementById("page-url").value = defaultUrl;

    fetchTasks();
    initializeCheckboxListeners();
    showDefinitions();
};