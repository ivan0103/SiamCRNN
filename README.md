<h1 align="center">Urban Change Detection Based on Remote Sensing Data</h1>
    <h2 align="center">How are Recurrent Neural Networks applied in the context of urban change detection?</h2>
    <h3 align="center"> This is an implementation of <strong>SiamCRNN</strong> framework forked from this official 
        <a href="https://github.com/ChenHongruixuan/SiamCRNN">repository</a>
    </h3>
    <h2>Description</h2>
    <p>
        This repository is part of my research, in collaboration with TU Delft and Gate Institute on "How are Recurrent Neural Networks applied in the context of urban change detection?". This repository has been forked, updated and modified. The model has been trained and evaluated on 4 different datasets - Levir, DSIFN, CDD and OSCD. The repository contains 4 additional branches for the implementation for each of these datasets. The branches have been updated so that they work inside a Google Colab environment with the dataset downloaded separately, modified with a script, and then added to the Google Colab environment through Google Drive.
    </p>
    <h2>Datasets</h2>
    <p>The model has been trained and evaluated on four different datasets:</p>
    <ul>
        <li>Levir-CD</li>
        <li>DSIFN</li>
        <li>CDD</li>
        <li>OSCD</li>
    </ul>
    <h2>Branches</h2>
    <p>The repository contains four branches, each dedicated to one of the datasets:</p>
    <ul>
        <li><strong>Levir-CD</strong>: For the Levir Change Detection dataset.</li>
        <li><strong>DSIFN</strong>: For the DSIFN dataset.</li>
        <li><strong>CDD</strong>: For the Change Detection Dataset.</li>
        <li><strong>OSCD</strong>: For the OSCD dataset.</li>
    </ul>
    <h2>Getting Started</h2>
    <h3>Prerequisites</h3>
    <ul>
        <li>Python 3.7+</li>
        <li>PyTorch</li>
        <li>Google Colab account (for running the model in the cloud)</li>
    </ul>
    <h3>Running in Google Colab</h3>
    <ol>
        <li><strong>Download the Dataset</strong>
            <p>Download the DSIFN dataset.</p>
        </li>
        <li><strong>Clone the Repository</strong>
            <pre><code>git clone https://github.com/yourusername/SiamCRNN.git
cd SiamCRNN</code></pre>
        </li>
        <li><strong>Switch to the Desired Branch</strong>
            <pre><code>git checkout DSIFN</code></pre>
        </li>
        <li><strong>Run Setup Script</strong>
            <p>Navigate to the script directory and modify the paths in <code>files.sh</code> if necessary:</p>
            <pre><code>cd FCN_version/script/
chmod +x files.sh
./files.sh</code></pre>
        </li>
        <li><strong>Upload Modified Dataset</strong>
            <p>Upload the modified dataset to your Google Drive in the following folder: MyDrive/Colab Notebooks</p>
        </li>
        <li><strong>Open Google Colab</strong>
            <p>Open the <code>GoogleColab.ipynb</code> file in Google Colab.</p>
        </li>
        <li><strong>Run the Notebook</strong>
            <p>Follow the instructions in the notebook to set up the environment and run the training and evaluation.</p>
        </li>
    </ol>
    <h2>Acknowledgments</h2>
    <ul>
        <li>TU Delft</li>
        <li>Gate Institute</li>
        <li>Authors of the <a href="https://github.com/ChenHongruixuan/SiamCRNN">original SiamCRNN repository</a></li>
    </ul>
    <p>For more detailed information on the research and implementation, please refer to the original paper and the documentation provided in this repository.</p>
