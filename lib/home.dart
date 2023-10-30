import 'dart:io';

import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as imglib;

import 'package:camera_app/main.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key}) : super(key: key);

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;

  // List? _result;
  List<List<double>>? _result;
  bool _imageSelected = false;
  bool _loading = false;
  bool _isDetected = false;
  final _imagePicker = ImagePicker();

  //---------------------------
  late tfl.Interpreter _interpreter;

  late List inputShape;
  late List outputShape;
  late tfl.TensorType inputType;
  late tfl.TensorType outputType;

  //-image specs---------------------------
  double x = 0;
  late double y = 0;
  late double h = 0;
  late double w = 0;
  late double cls = 0;
  late double conf = 0;

  //--camera----------------------------------
  CameraImage? cameraImages;
  CameraController? cameraController;

  // bool _loadingPredictions = false;

  //----batch detections---------------------------------------
  // List<File> batch = [];
  // List<List<List<double>>> _batchResults = [];

  //-------------------------------------
  //--------------------------image selection----------------------------
  @override
  Future getImage(ImageSource source) async {
    final image = await ImagePicker().pickImage(source: source);
    if (image == null) {
      return;
    }
    final imageTemporary = File(image.path);
    setState(() {
      _image = imageTemporary;
      _imageSelected = false;
      _result = null;
    });
    classifyImage(_image);
  }

  //-----------------------ML-------------------------------------------------
  //----for image---------------

  Future classifyImage(File? image) async {
    if (image == null) {
      return;
    }
    final imageBytes = await image.readAsBytes();

    var inputTensor = preProcessImage(imageBytes);
    var outputTensor = List.filled(1 * 3087 * 6, 0.0).reshape([1, 3087, 6]);
    _interpreter.run(inputTensor, outputTensor);

    List<List<double>> detections = postProcess(outputTensor);
    print("output tensor :$outputTensor");
    print(detections);
    setState(() {
      if (detections.isEmpty) {
        conf = 0;
      } else {
        conf = detections[0][4];
      }
      _loading = false;
      _isDetected = true;
      _result = detections;
    });
  }

  //--------for video-----------------------------------------------------------------------
  void classifyVideo(CameraImage image) async {
    print("camera open");
    //preprocessing video-----------------------------
    List<double> pixels = [];
    for (var plane in image.planes) {
      pixels.addAll(plane.bytes.map((byte) => byte.toDouble()));
    }
    List<List<List<List<double>>>> inputTensor = [
      List.generate(224, (row) {
        return List.generate(224, (col) {
          double r = pixels[row * 224 + col];
          double g = pixels[224 * 224 + row * 224 + col];
          double b = pixels[2 * 224 * 224 + row * 224 + col];
          return [r / 255.0, g / 255.0, b / 255.0];
        });
      })
    ];

    var outputTensor = List.filled(1 * 3087 * 6, 0.0).reshape([1, 3087, 6]);
    //-----------------------------------------------------
    _interpreter.run(inputTensor, outputTensor);
    List<List<double>> detections = postProcess(outputTensor);
    //print("------output detection best video-----------$detections");

    setState(() {
      if (detections.isNotEmpty) {
        conf = 1;
        _loading = false;
        _result = detections;
      }
    });
  }

  //-------------------------------------------------------------------------------------------------
//--------------------------------data processing-----------------------------------------------------------------
//-------------------------------image processing---------------------------------------------------
  List<List<double>> postProcess(List<dynamic> outputTensor) {
    double maxConfidence = 0.3; //threshhold
    // double iou_threshold = 0.9;
    List<List<double>> detections = [];
    for (int i = 0; i < outputTensor[0].length; i++) {
      List<dynamic> prediction = outputTensor[0][i];
      double x = prediction[0];
      double y = prediction[1];
      double w = prediction[2];
      double h = prediction[3];
      double conf = prediction[4];

      if (conf > maxConfidence) {
        detections.add([x, y, w, h, conf, prediction[5]]);
      }
    }

    detections.sort((a, b) => b[4].compareTo(a[4]));
    print("detections passed the threshold :${detections.length}");

    print('--------------------------Detections Array: $detections');
    return detections;
  }

  List<List<List<List<double>>>> preProcessImage(Uint8List imageBytes) {
    imglib.Image img = imglib.decodeImage(imageBytes)!;
    imglib.Image resizedImage = imglib.copyResize(img, width: 224, height: 224);

    List<List<List<List<double>>>> inputValues = List.generate(1, (batchIndex) {
      List<List<List<double>>> batch = [];
      for (int row = 0; row < 224; row++) {
        List<List<double>> rowValues = [];
        for (int col = 0; col < 224; col++) {
          List<double> pixelValues = [];

          int pixel = resizedImage.getPixel(col, row);
          double r = imglib.getRed(pixel) / 255.0;
          double g = imglib.getGreen(pixel) / 255.0;
          double b = imglib.getBlue(pixel) / 255.0;

          pixelValues.add(r);
          pixelValues.add(g);
          pixelValues.add(b);

          rowValues.add(pixelValues);
        }
        batch.add(rowValues);
      }
      return batch;
    });

    return inputValues;
  }

//---------------------------------------------------------------------------------------------------
  bool saving = false;
  bool processing = false;

//-------------------------------------------------------------
  // Input shape: [1, 224, 224, 3]
  // Output shape: [1, 10647, 6]

  loadCamera() {
    cameraController = CameraController(cameras![0], ResolutionPreset.medium);
    cameraController!.initialize().then((value) {
      if (!mounted) {
        return;
      } else {
        setState(() {
          cameraController!.startImageStream((imageStream) {
            cameraImages = imageStream;
            //classifyVideo(imageStream);
          });
        });
      }
    });
  }

  Future<void> loadModel() async {
    _interpreter = await tfl.Interpreter.fromAsset("assets/chicken.tflite");
    inputShape = _interpreter.getInputTensor(0).shape;
    outputShape = _interpreter.getOutputTensor(0).shape;
    print('--------------------------Input shape: $inputShape');
    print('--------------------------Output shape: $outputShape');

    // print(
    //     '--------------------Input shape: ${_classifier.getInputTensor(0).shape}');
    // print(
    //     '--------------------Output shape: ${_classifier.getOutputTensor(0).shape}');
    inputType = _interpreter.getInputTensor(0).type;
    outputType = _interpreter.getOutputTensor(0).type;
    print('--------------------------Input type: $inputType');
    print('--------------------------Output type: $outputType');
  }

  void initState() {
    super.initState();
    _loading = true;
    // UserSheetsApi.init();
    loadModel().then((value) {
      setState(() {
        _loading = false;
      });
    });
  }
  // @override
  // Widget build(BuildContext context) {
  //   return Scaffold(
  //     appBar: AppBar(
  //       title: const Text('Chicken Detector'),
  //     ),
  //     body: Center(
  //       child: Column(
  //         children: [
  //           if (_image != null)
  //             Stack(
  //               children: [
  //                 Image.file(
  //                   _image!,
  //                   width: 224,
  //                   height: 224,
  //                   fit: BoxFit.cover,
  //                 ),
  //                 if (_result != null)
  //                   Positioned.fill(
  //                     child: CustomPaint(
  //                       painter: BoundingBoxPainter(
  //                         imageSize: const Size(224, 224),
  //                         detection: _result!,
  //                       ),
  //                     ),
  //                   ),
  //               ],
  //             ),
  //           CustomButton('Pick from Gallery', () => getImage(ImageSource.gallery)),
  //         ],
  //       ),
  //     ),
  //   );
  // }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chicken Detector'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_result != null)
              Positioned.fill(
                child: CustomPaint(
                  painter: BoundingBoxPainter(
                    imageSize: const Size(224, 224),
                    detection: _result!,
                  ),
                ),
              ),
            if (_image != null)
              Image.file(
                _image!,
                width: 224,
                height: 224,
                fit: BoxFit.cover,
              ),
            CustomButton(
                'Pick from Gallery', () => getImage(ImageSource.gallery)),
          ],
        ),
      ),
    );
  }
}

class CustomButton extends StatelessWidget {
  final String title;
  final VoidCallback onClick;

  const CustomButton(this.title, this.onClick, {super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 280,
      child: ElevatedButton(
        onPressed: onClick,
        child: Align(
          alignment: Alignment.center,
          child: Text(title),
        ),
      ),
    );
  }
}

//--------------------bounding boxes------------------------
void drawBoundingBox(
    Canvas canvas, Size imageSize, List<List<double>> detections) {
  for (var detection in detections) {
    double x = detection[0];
    double y = detection[1];
    double w = detection[2];
    double h = detection[3];
    double confidence = detection[4];

    if (confidence >= 0.3) {
      // Scale the coordinates to match the image dimensions
      double imageWidth = imageSize.width;
      double imageHeight = imageSize.height;

      x *= imageWidth;
      y *= imageHeight;
      w *= imageWidth;
      h *= imageHeight;

      double left = x - w / 2;
      double top = y - h / 2;
      double right = x + w / 2;
      double bottom = y + h / 2;

      // Create a paint object to define the bounding box style
      Paint paint = Paint()
        ..color = Colors.green
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;

      canvas.drawRect(Rect.fromLTRB(left, top, right, bottom), paint);

      //text
      TextStyle textStyle = const TextStyle(
        color: Colors.white,
        fontSize: 16.0,
        fontWeight: FontWeight.bold,
        backgroundColor: Colors.green,
      );
      TextSpan textSpan = TextSpan(
        text: '${(confidence * 100).toStringAsFixed(2)}%',
        style: textStyle,
      );
      TextPainter textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      double textX = left;
      double textY = top - 20.0;

      textPainter.paint(canvas, Offset(textX, textY));
    } else {
      print("No detections");
    }
  }
}

class BoundingBoxPainter extends CustomPainter {
  final Size imageSize;
  final List<List<double>> detection;

  BoundingBoxPainter({
    required this.imageSize,
    required this.detection,
  });

  @override
  void paint(Canvas canvas, Size size) {
    drawBoundingBox(canvas, imageSize, detection);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return false;
  }
}
