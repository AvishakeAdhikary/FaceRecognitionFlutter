import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:camera/camera.dart';
import 'package:firebase_ml_vision/firebase_ml_vision.dart';

import 'dart:convert';
import 'dart:io';

import 'detector_painters.dart';
import 'utilities.dart';

import 'package:quiver/collection.dart';
import 'package:flutter/services.dart';

import 'package:image/image.dart' as imageLibrary;
import 'package:tflite_flutter/tflite_flutter.dart' as tensorflowlite;

final String appName = "Face Recognition Flutter";

void main() 
{
  runApp(MaterialApp(
    themeMode: ThemeMode.light,
    theme: ThemeData(
      brightness: Brightness.light,
      primaryColor: Colors.red[800],
      accentColor: Colors.redAccent[400],
      ),
    home: FaceRecognitionFlutter(),
    title: appName,
    debugShowCheckedModeBanner: false,
  ));
}

class FaceRecognitionFlutter extends StatefulWidget 
{
  @override
  FaceRecognitionFlutterState createState() => FaceRecognitionFlutterState();
}

class FaceRecognitionFlutterState extends State<FaceRecognitionFlutter> 
{
 
  File jsonFile;
  Directory temporaryDirectory;
  CameraLensDirection _cameraLensDirection = CameraLensDirection.front;
  CameraController _cameraController;
  var interpreter;
  dynamic _scanResults;

  dynamic data = {};
  List e1;
  double threshold = 1.0;

  final TextEditingController _userName = new TextEditingController();
  
  bool _isFaceFound = false;
  bool _isDetecting = false;
  
  @override
  void initState() {
    super.initState();

    SystemChrome.setPreferredOrientations(
        [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);
    _initializeCamera();
  }

  Future loadTFLiteModel() async {
    try {
      final gpuDelegateV2 = tensorflowlite.GpuDelegateV2(
          options: tensorflowlite.GpuDelegateOptionsV2(
        false,
        tensorflowlite.TfLiteGpuInferenceUsage.fastSingleAnswer,
        tensorflowlite.TfLiteGpuInferencePriority.minLatency,
        tensorflowlite.TfLiteGpuInferencePriority.auto,
        tensorflowlite.TfLiteGpuInferencePriority.auto,
      ));

      var interpreterOptions = tensorflowlite.InterpreterOptions()
        ..addDelegate(gpuDelegateV2);
      interpreter = await tensorflowlite.Interpreter.fromAsset('mobilefacenet.tflite',
          options: interpreterOptions);
    } on Exception {
      print('Failed To Load TensorFlow Model.');
    }
  }

  void _initializeCamera() async {
    await loadTFLiteModel();
    CameraDescription cameraDescription = await getCamera(_cameraLensDirection);

    ImageRotation rotation = rotationIntToImageRotation(
      cameraDescription.sensorOrientation,
    );

    _cameraController =
        CameraController(cameraDescription, ResolutionPreset.low, enableAudio: false);
    await _cameraController.initialize();
    await Future.delayed(Duration(milliseconds: 500));
    temporaryDirectory = await getApplicationDocumentsDirectory();
    String _embPath = temporaryDirectory.path + '/emb.json';
    jsonFile = new File(_embPath);
    if (jsonFile.existsSync()) data = json.decode(jsonFile.readAsStringSync());

    _cameraController.startImageStream((CameraImage image) {
      if (_cameraController != null) {
        if (_isDetecting) return;
        _isDetecting = true;
        String res;
        dynamic finalResult = Multimap<String, Face>();
        detect(image, _getDetectionMethod(), rotation).then(
          (dynamic result) async {
            if (result.length == 0)
              _isFaceFound = false;
            else
              _isFaceFound = true;
            Face _face;
            imageLibrary.Image convertedImage =
                _convertCameraImage(image, _cameraLensDirection);
            for (_face in result) {
              double x, y, w, h;
              x = (_face.boundingBox.left - 10);
              y = (_face.boundingBox.top - 10);
              w = (_face.boundingBox.width + 10);
              h = (_face.boundingBox.height + 10);
              imageLibrary.Image croppedImage = imageLibrary.copyCrop(
                  convertedImage, x.round(), y.round(), w.round(), h.round());
              croppedImage = imageLibrary.copyResizeCropSquare(croppedImage, 112);
              // int startTime = new DateTime.now().millisecondsSinceEpoch;
              res = _recognize(croppedImage);
              // int endTime = new DateTime.now().millisecondsSinceEpoch;
              // print("Inference took ${endTime - startTime}ms");
              finalResult.add(res, _face);
            }
            setState(() {
              _scanResults = finalResult;
            });

            _isDetecting = false;
          },
        ).catchError(
          (_) {
            _isDetecting = false;
          },
        );
      }
    });
  }

  HandleDetection _getDetectionMethod() {
    final faceDetector = FirebaseVision.instance.faceDetector(
      FaceDetectorOptions(
        mode: FaceDetectorMode.accurate,
      ),
    );
    return faceDetector.processImage;
  }

  Widget _buildResults() {
    const Text noResultsText = const Text('');
    if (_scanResults == null ||
        _cameraController == null ||
        !_cameraController.value.isInitialized) {
      return noResultsText;
    }
    CustomPainter painter;

    final Size imageSize = Size(
      _cameraController.value.previewSize.height,
      _cameraController.value.previewSize.width,
    );
    painter = FaceDetectorPainter(imageSize, _scanResults);
    return CustomPaint(
      painter: painter,
    );
  }

  Widget _buildImage() {
    if (_cameraController == null || !_cameraController.value.isInitialized) {
      return Center(
        child: CircularProgressIndicator(),
      );
    }

    return Container(
      constraints: const BoxConstraints.expand(),
      child: _cameraController == null
          ? const Center(child: null)
          : Stack(
              fit: StackFit.expand,
              children: <Widget>[
                CameraPreview(_cameraController),
                _buildResults(),
              ],
            ),
    );
  }

  void _toggleCameraDirection() async {
    if (_cameraLensDirection == CameraLensDirection.back) {
      _cameraLensDirection = CameraLensDirection.front;
    } else {
      _cameraLensDirection = CameraLensDirection.back;
    }
    await _cameraController.stopImageStream();
    await _cameraController.dispose();

    setState(() {
      _cameraController = null;
    });

    _initializeCamera();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(appName),
        actions: <Widget>[
          PopupMenuButton<Choice>(
            onSelected: (Choice result) {
              if (result == Choice.delete)
                _resetFile();
              else
                _viewLabels();
            },
            itemBuilder: (BuildContext context) => <PopupMenuEntry<Choice>>[
              const PopupMenuItem<Choice>(
                child: Text('View Saved Faces'),
                value: Choice.view,
              ),
              const PopupMenuItem<Choice>(
                child: Text('Remove All Faces'),
                value: Choice.delete,
              )
            ],
          ),
        ],
      ),
      body: _buildImage(),
      floatingActionButton:
          Column(mainAxisAlignment: MainAxisAlignment.end, children: [
        FloatingActionButton(
          backgroundColor: (_isFaceFound) ? Colors.red[800] : Colors.redAccent[400],
          child: Icon(Icons.add),
          onPressed: () {
            if (_isFaceFound) _addLabel();
          },
          heroTag: null,
        ),
        SizedBox(
          height: 10,
        ),
        FloatingActionButton(
          onPressed: _toggleCameraDirection,
          heroTag: null,
          child: _cameraLensDirection == CameraLensDirection.back
              ? const Icon(Icons.camera_front)
              : const Icon(Icons.camera_rear),
        ),
      ]),
    );
  }

  imageLibrary.Image _convertCameraImage(
      CameraImage image, CameraLensDirection _dir) {
    int width = image.width;
    int height = image.height;
    // imageLibrary -> Image package from https://pub.dartlang.org/packages/image
    var img = imageLibrary.Image(width, height); // Create Image buffer
    const int hexFF = 0xFF000000;
    final int uvyButtonStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel;
    for (int x = 0; x < width; x++) 
    {
      for (int y = 0; y < height; y++) 
      {
        final int uvIndex = uvPixelStride * (x / 2).floor() + uvyButtonStride * (y / 2).floor();
        final int index = y * width + x;
        final yp = image.planes[0].bytes[index];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];
        // Calculate pixel color
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
            .round()
            .clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
        // color: 0x FF  FF  FF  FF
        //           A   B   G   R
        img.data[index] = hexFF | (b << 16) | (g << 8) | r;
      }
    }
    var img1 = (_dir == CameraLensDirection.front)
        ? imageLibrary.copyRotate(img, -90)
        : imageLibrary.copyRotate(img, 90);
    return img1;
  }

  String _recognize(imageLibrary.Image img) 
  {
    List input = imageToByteListFloat32(img, 112, 128, 128);
    input = input.reshape([1, 112, 112, 3]);
    List output = List(1 * 192).reshape([1, 192]);
    interpreter.run(input, output);
    output = output.reshape([192]);
    e1 = List.from(output);
    return _compare(e1).toUpperCase();
  }

  String _compare(List currEmb) 
  {
    if (data.length == 0) return "No Face saved";
    double minDist = 999;
    double currDist = 0.0;
    String predRes = "NOT RECOGNIZED";
    for (String label in data.keys) 
    {
      currDist = euclideanDistance(data[label], currEmb);
      if (currDist <= threshold && currDist < minDist) 
      {
        minDist = currDist;
        predRes = label;
      }
    }
    print(minDist.toString() + " " + predRes);
    return predRes;
  }

  void _resetFile() 
  {
    data = {};
    jsonFile.deleteSync();
  }

  void _viewLabels() 
  {
    setState(() {
      _cameraController = null;
    });
    String name;
    var alert = new AlertDialog(
      title: new Text("Saved Faces"),
      content: new ListView.builder(
          padding: new EdgeInsets.all(2),
          itemCount: data.length,
          itemBuilder: (BuildContext context, int index) {
            name = data.keys.elementAt(index);
            return new Column(
              children: <Widget>[
                new ListTile(
                  title: new Text(
                    name,
                    style: new TextStyle(
                      fontSize: 14,
                      color: Colors.red[400],
                    ),
                  ),
                ),
                new Padding(
                  padding: EdgeInsets.all(2),
                ),
                new Divider(),
              ],
            );
          }),
      actions: <Widget>[
        new FlatButton(
          child: Text("OK"),
          onPressed: () {
            _initializeCamera();
            Navigator.pop(context);
          },
        )
      ],
    );
    showDialog(
        context: context,
        builder: (context) {
          return alert;
        });
  }

  void _addLabel() {
    setState(() {
      _cameraController = null;
    });
    print("Adding New Face");
    var alert = new AlertDialog(
      title: new Text("Add Face"),
      content: new Row(
        children: <Widget>[
          new Expanded(
            child: new TextField(
              controller: _userName,
              autofocus: true,
              decoration: new InputDecoration(
                  labelText: "Name", icon: new Icon(Icons.face)),
            ),
          )
        ],
      ),
      actions: <Widget>[
        new FlatButton(
            child: Text("Save"),
            onPressed: () {
              _handle(_userName.text.toUpperCase());
              _userName.clear();
              Navigator.pop(context);
            }),
        new FlatButton(
          child: Text("Cancel"),
          onPressed: () {
            _initializeCamera();
            Navigator.pop(context);
          },
        )
      ],
    );
    showDialog(
        context: context,
        builder: (context) {
          return alert;
        });
  }

  void _handle(String text) 
  {
    data[text] = e1;
    jsonFile.writeAsStringSync(json.encode(data));
    _initializeCamera();
  }
}
