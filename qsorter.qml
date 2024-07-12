import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    visible: true
    width: 800
    height: 600
    title: "Image Sorter"

    ColumnLayout {
        anchors.fill: parent
        spacing: 5

        // Верхняя секция (содержит (1), изображение, (2))
        SplitView {
            Layout.fillWidth: true
            Layout.preferredHeight: 200
            orientation: Qt.Horizontal

            Rectangle {
                border.color: "red"
                border.width: 2
                SplitView.preferredWidth: 200
                Text {
                    anchors.centerIn: parent
                    text: "(1)"
                    font.pixelSize: 24
                    color: "red"
                }
            }

            Rectangle {
                border.color: "red"
                border.width: 2
                SplitView.fillWidth: true
                Image {
                    id: imageLabel
                    anchors.fill: parent
                    source: currentImagePath
                    fillMode: Image.PreserveAspectFit
                }
            }

            Rectangle {
                border.color: "red"
                border.width: 2
                SplitView.preferredWidth: 200
                Text {
                    anchors.centerIn: parent
                    text: "(2)"
                    font.pixelSize: 24
                    color: "red"
                }
            }
        }

        // Разделитель между верхней секцией и кнопками
        Splitter {
            id: splitter1
            orientation: Qt.Vertical
            Layout.fillWidth: true
            height: 10
            Rectangle {
                height: 1
                width: parent.width
                color: "gray"
                anchors.centerIn: parent
            }
        }

        // Полоса с кнопками
        RowLayout {
            Layout.fillWidth: true
            spacing: 5

            Repeater {
                model: ["Ruki (1)", "Perchatki (2)", "Drugoe (3)", "Load Images"]
                delegate: Button {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 40
                    text: modelData
                    onClicked: {
                        if (index < 3) {
                            sorter.sortImage(["Ruki", "Perchatki", "Drugoe"][index])
                        } else {
                            sorter.loadImages()
                        }
                    }
                }
            }
        }

        // Разделитель между кнопками и нижней секцией
        Splitter {
            id: splitter2
            orientation: Qt.Vertical
            Layout.fillWidth: true
            height: 10
            Rectangle {
                height: 1
                width: parent.width
                color: "gray"
                anchors.centerIn: parent
            }
        }

        // Нижняя секция (3)
        SplitView {
            Layout.fillWidth: true
            orientation: Qt.Vertical
            height: 100

            Rectangle {
                Layout.fillWidth: true
                SplitView.fillHeight: true
                border.color: "red"
                border.width: 2
                Text {
                    anchors.centerIn: parent
                    text: "(3)"
                    font.pixelSize: 24
                    color: "red"
                }
            }
        }
    }
}
