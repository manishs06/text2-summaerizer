document.addEventListener('DOMContentLoaded', () => {
    const inputText = document.getElementById('input-text');
    const charCount = document.getElementById('char-count');
    const summarizeBtn = document.getElementById('summarize-btn');
    const methodRadios = document.getElementsByName('method');
    const extractiveOptions = document.getElementById('extractive-options');
    const abstractiveOptions = document.getElementById('abstractive-options');
    
    // Sliders
    const numSentences = document.getElementById('num_sentences');
    const sentenceVal = document.getElementById('sentence-val');
    const maxLength = document.getElementById('max_length');
    const maxLenVal = document.getElementById('max-len-val');
    const minLength = document.getElementById('min_length');
    const minLenVal = document.getElementById('min-len-val');
    
    // Container and results
    const resultPlaceholder = document.querySelector('.placeholder-content');
    const loading = document.getElementById('loading');
    const summaryContent = document.getElementById('summary-content');
    const summaryText = document.getElementById('summary-text');
    const copyBtn = document.getElementById('copy-btn');
    const statsGrid = document.getElementById('stats');
    
    // Stats elements
    const statReduction = document.getElementById('stat-reduction');
    const statOriginal = document.getElementById('stat-original');
    const statSummary = document.getElementById('stat-summary');

    // Update character count
    inputText.addEventListener('input', () => {
        charCount.textContent = `${inputText.value.length} characters`;
    });

    // Toggle options based on method
    methodRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            if (radio.value === 'extractive') {
                extractiveOptions.classList.remove('hidden');
                abstractiveOptions.classList.add('hidden');
            } else {
                extractiveOptions.classList.add('hidden');
                abstractiveOptions.classList.remove('hidden');
            }
        });
    });

    // Slider value updates
    numSentences.addEventListener('input', () => sentenceVal.textContent = numSentences.value);
    maxLength.addEventListener('input', () => {
        maxLenVal.textContent = maxLength.value;
        if (parseInt(maxLength.value) < parseInt(minLength.value)) {
            minLength.value = maxLength.value;
            minLenVal.textContent = maxLength.value;
        }
    });
    minLength.addEventListener('input', () => {
        minLenVal.textContent = minLength.value;
        if (parseInt(minLength.value) > parseInt(maxLength.value)) {
            maxLength.value = minLength.value;
            maxLenVal.textContent = minLength.value;
        }
    });

    // Copy to clipboard
    copyBtn.addEventListener('click', () => {
        if (summaryText.textContent) {
            navigator.clipboard.writeText(summaryText.textContent).then(() => {
                const icon = copyBtn.querySelector('i');
                icon.classList.remove('fa-copy');
                icon.classList.add('fa-check');
                setTimeout(() => {
                    icon.classList.remove('fa-check');
                    icon.classList.add('fa-copy');
                }, 2000);
            });
        }
    });

    // Summarize action
    summarizeBtn.addEventListener('click', async () => {
        const text = inputText.value.trim();
        if (text.length < 10) {
            alert('Please enter at least 10 characters.');
            return;
        }

        const method = document.querySelector('input[name="method"]:checked').value;
        const payload = {
            text: text,
            method: method
        };

        if (method === 'extractive') {
            payload.algorithm = document.getElementById('algorithm').value;
            payload.num_sentences = parseInt(numSentences.value);
        } else {
            payload.max_length = parseInt(maxLength.value);
            payload.min_length = parseInt(minLength.value);
        }

        // UI State: Loading
        resultPlaceholder.classList.add('hidden');
        summaryContent.classList.add('hidden');
        statsGrid.classList.add('hidden');
        loading.classList.remove('hidden');
        summarizeBtn.disabled = true;
        summarizeBtn.style.opacity = '0.7';

        try {
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // UI State: Result
            summaryText.textContent = data.summary;
            statReduction.textContent = `${data.reduction}%`;
            statOriginal.textContent = data.original_length;
            statSummary.textContent = data.summary_length;

            loading.classList.add('hidden');
            summaryContent.classList.remove('hidden');
            statsGrid.classList.remove('hidden');
        } catch (error) {
            alert(`Error: ${error.message}`);
            resultPlaceholder.classList.remove('hidden');
            loading.classList.add('hidden');
        } finally {
            summarizeBtn.disabled = false;
            summarizeBtn.style.opacity = '1';
        }
    });
});
