// static/js/main.js

// --- PART 1: LIVE STATUS POLLING LOGIC ---
let wasRunning = false;

async function fetchStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        const statusText = document.getElementById('status-text');
        const statusDetails = document.getElementById('status-details');
        const trainingFieldset = document.getElementById('training-fieldset');
        const cancelForm = document.getElementById('cancel-form');
        const progressContainer = document.getElementById('progress-container');
        const statusBox = document.getElementById('status-box');

        statusText.textContent = data.status;
        statusText.className = `status-${data.status.toLowerCase()}`;
        statusBox.setAttribute('data-status', data.status);
        
        if (data.status === 'Running') {
            if (!wasRunning) { statusDetails.textContent = data.details; }
            trainingFieldset.disabled = true;
            cancelForm.style.display = 'block';
            progressContainer.style.display = 'block';
            wasRunning = true;
        } else { // Idle
            if (wasRunning) {
                if (data.message) {
                    const category = data.details.includes('successfully') ? 'success' : 'error';
                    window.location.href = `/?message=${encodeURIComponent(data.message)}&category=${category}`;
                } else {
                    window.location.reload();
                }
            }
            statusDetails.textContent = data.details;
            trainingFieldset.disabled = false;
            cancelForm.style.display = 'none';
            progressContainer.style.display = 'none';
            wasRunning = false;
        }
    } catch (error) { console.error('Error fetching status:', error); }
}

// --- PART 2: EVENT LISTENERS AND INITIALIZATION ---
document.addEventListener('DOMContentLoaded', function() {
    // A. Start the live status polling
    setInterval(fetchStatus, 3000);
    fetchStatus();

    // B. Set up the click handlers for the sample images
    const sampleImages = document.querySelectorAll('.sample-image');
    const modelSelect = document.getElementById('model_path');
    const fileInput = document.getElementById('image_file');

    sampleImages.forEach(image => {
        image.addEventListener('click', function() {
            const selectedModel = modelSelect.value;
            const selectedImageName = this.dataset.filename;
            if (!selectedModel) {
                alert('Please select a trained model first.');
                return;
            }
            const predictUrl = `/predict_sample?model_path=${encodeURIComponent(selectedModel)}&image_name=${encodeURIComponent(selectedImageName)}`;
            window.location.href = predictUrl;
        });
    });

    // C. Add logic to clear selections for a better user experience
    fileInput.addEventListener('click', function() {
         document.querySelectorAll('.sample-image.selected').forEach(img => img.classList.remove('selected'));
    });
    
    sampleImages.forEach(image => {
        image.addEventListener('click', function() {
            fileInput.value = '';
            document.querySelectorAll('.sample-image.selected').forEach(img => img.classList.remove('selected'));
            this.classList.add('selected');
        });
    });
});