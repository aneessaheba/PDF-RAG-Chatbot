const API_URL = 'http://localhost:8000';

async function uploadPDF() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a PDF file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('uploadStatus').innerHTML = 'Uploading...';

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        document.getElementById('uploadStatus').innerHTML =
            `<div class="result">${data.message || 'Upload successful!'}</div>`;
    } catch (error) {
        document.getElementById('uploadStatus').innerHTML =
            `<div class="result">Error: ${error.message}</div>`;
    }
}

async function askQuestion() {
    const question = document.getElementById('questionInput').value;

    if (!question) {
        alert('Please enter a question');
        return;
    }

    document.getElementById('queryResult').innerHTML = 'Thinking...';

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();

        let html = `<div class="result"><strong>Answer:</strong><p>${data.answer}</p>`;

        if (data.sources) {
            html += '<strong>Sources:</strong>';
            data.sources.forEach(source => {
                html += `<div class="source">
                    ${source.file} (Page ${source.page})<br>
                    <small>${source.content}</small>
                </div>`;
            });
        }
        html += '</div>';

        document.getElementById('queryResult').innerHTML = html;
    } catch (error) {
        document.getElementById('queryResult').innerHTML =
            `<div class="result">Error: ${error.message}</div>`;
    }
}
