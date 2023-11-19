const express = require('express');
const multer = require('multer');
const path = require('path');

const app = express();
const port = 3000;


const storage = multer.diskStorage({
    destination: 'uploads/',  
    filename: function (req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
    }
});

const upload = multer({ storage: storage });



app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'home.html'));
});


app.use(express.static(__dirname));

app.post('/upload', upload.single('image'), (req, res) => {
    if (req.file) {
        res.json({ message: 'Image uploaded successfully!' });
    } else {
        res.status(400).json({ message: 'Error uploading image.' });
    }
});


app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});
