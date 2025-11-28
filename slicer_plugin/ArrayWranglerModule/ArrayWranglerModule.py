import logging
import os
import vtk

from PythonQt.QtCore import Qt

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)

from PythonQt.QtGui import (
    # QAbstractTableModel,
    # QTableWidgetItem,
    QTreeWidgetItem,
)

from slicer.util import VTKObservationMixin

# pylint: skip-file

from sort_library import sorting_logic as slogic
from scipy import ndimage as ndi
import numpy as np
import random
import skimage as ski

logging.basicConfig(level=logging.DEBUG)

# pylint: disable=E1101

#
# ArrayWranglerModule
#

# Custom attribute on folder objects,
# indicating that it contains a dataset
# the affirmative value is a string 'True',
# otherwise, whe folder doesn't have this attribute
# the polling function returns an emtry string
FOLDER_ATTRIBUTE = "IsDataset"
FOLDER_ATTRIBUTE_VALUE = "True"


class ArrayWranglerModule(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "ArrayWranglerModule"
        self.parent.categories = [
            "Examples"
        ]  # TODO: set categories (folders where the module is in the module selector)
        self.parent.dependencies = (
            []
        )  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "John Doe (AnyWare Corp.)"
        ]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module
        self.parent.helpText = """

"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
Originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.,
Andras Lasso, PerkLab, and Steve Pieper, Isomics, Inc.
and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users
    # to make it easy to try the module,
    # but if no sample data is available then this method
    # (and associated startupCompeted signal connection) can be removed.


#
# ArrayWranglerModuleWidget
#


class ArrayWranglerModuleWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time
        and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time
        and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(
            self.resourcePath("UI/ArrayWranglerModule.ui")
        )  # pylint: disable=no-member
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(
            uiWidget
        )  # pylint: disable=no-member

        # Set scene in MRML widgets. Make sure that in Qt designer the
        # top-level qMRMLWidget's "mrmlSceneChanged(vtkMRMLScene*)" signal
        # in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations
        # that should be possible to run in batch mode,
        # without a graphical user interface.
        self.logic = ArrayWranglerModuleLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # These connections ensure that whenever user changes some settings on the GUI,
        # that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )  # pylint: disable=no-member
        self.ui.inputSegmentSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )  # pylint: disable=no-member
        self.ui.outputSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )  # pylint: disable=no-member
        self.ui.alignHelperSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )  # pylint: disable=no-member
        # self.ui.imageNumLayersSliderWidget.connect(
        #     "valueChanged(double)", self.updateParameterNodeFromGUI
        # )
        self.ui.saveField.connect(
            "currentPathChanged(QString)", self.updateParameterNodeFromGUI
        )  # not working, TBD
        self.ui.savePathSegm.connect(
            "currentPathChanged(QString)", self.updateParameterNodeFromGUI
        )  # not working, TBD
        # self.ui.checkPreserve.connect(
        #     "stateChanged(int)", self.updateParameterNodeFromGUI
        # )
        self.ui.checkRasCompatible.connect(
            "stateChanged(int)", self.updateParameterNodeFromGUI
        )

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.assessButton.connect("clicked(bool)", self.onAssessButton)
        self.ui.binaryErosionButton.connect("clicked(bool)", self.onErosionButton)
        self.ui.removeSmallObjectsButton.connect("clicked(bool)", self.onRemoveObButton)
        self.ui.breakButton.connect("clicked(bool)", self.onBreakButton)
        self.ui.exportButton.connect("clicked(bool)", self.onExportButton)
        self.ui.activateHelperButton.connect(
            "clicked(bool)", self.onActivateHelperButton
        )
        self.ui.btflip0.connect("clicked(bool)", self.onFlip0Button)
        self.ui.btflip1.connect("clicked(bool)", self.onFlip1Button)
        self.ui.btflip2.connect("clicked(bool)", self.onFlip2Button)
        self.ui.btswap01.connect("clicked(bool)", self.onSwap01Button)
        self.ui.btswap02.connect("clicked(bool)", self.onSwap02Button)
        self.ui.btswap12.connect("clicked(bool)", self.onSwap12Button)
        self.ui.btRotYC.connect("clicked(bool)", self.onRotYCButton)
        self.ui.btRotXC.connect("clicked(bool)", self.onRotXCButton)
        self.ui.btRotZC.connect("clicked(bool)", self.onRotZCButton)
        self.ui.btRotYCC.connect("clicked(bool)", self.onRotYCCButton)
        self.ui.btRotXCC.connect("clicked(bool)", self.onRotXCCButton)
        self.ui.btRotZCC.connect("clicked(bool)", self.onRotZCCButton)
        self.ui.refreshLocalButton.connect("clicked(bool)", self.onRefreshLocalButton)
        self.ui.treeDatasetLocal.setColumnCount(1)
        self.ui.treeDatasetLocal.setHeaderLabels(["Name"])
        self.ui.treeDatasetLocal.clicked.connect(self.onDatasetAvailClicked)
        self.ui.evaluateMaxDimButton.connect(
            "clicked(bool)", self.onEvaluateMaxDimButton
        )
        self.ui.newSizeButton.connect("clicked(bool)", self.onSetNewShapeButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes
        # (GUI wlil be updated when the user enters into the module)
        self.removeObserver(
            self._parameterNode,
            vtk.vtkCommand.ModifiedEvent,
            self.updateGUIFromParameterNode,
        )

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed
        # then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values,
        # node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks
        # for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass(
                "vtkMRMLScalarVolumeNode"
            )
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID(
                    "InputVolume", firstVolumeNode.GetID()
                )
        self._parameterNode.SetParameter(
            "RasCompatible", str(self.ui.checkRasCompatible.checked)
        )

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed
        then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer
        # to the newly selected.
        # Changes of parameter node are observed so that
        # whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI
        # (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputSelector.setCurrentNode(
            self._parameterNode.GetNodeReference("InputVolume")
        )
        self.ui.inputSegmentSelector.setCurrentNode(
            self._parameterNode.GetNodeReference("InputSementation")
        )
        self.ui.outputSelector.setCurrentNode(
            self._parameterNode.GetNodeReference("OutputEnum")
        )
        self.ui.alignHelperSelector.setCurrentNode(
            self._parameterNode.GetNodeReference("alignHelperVolume")
        )
        # self.ui.imageNumLayersSliderWidget.value = int(
        #     self._parameterNode.GetParameter("NumLayers")
        # )
        self.ui.saveField.currentPath = self._parameterNode.GetParameter(
            "SavePathSource"
        )  # not working, TBD
        self.ui.savePathSegm.currentPath = self._parameterNode.GetParameter(
            "SavePathSegm"
        )  # not working, TBD

        # Update buttons states and tooltips
        if (
            self._parameterNode.GetNodeReference("InputVolume")
            and self._parameterNode.GetNodeReference("InputSementation")
            and self._parameterNode.GetNodeReference("OutputEnum")
        ):
            self.ui.applyButton.toolTip = "Compute output volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input and output volume nodes"
            self.ui.applyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node
        (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = (
            self._parameterNode.StartModify()
        )  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID(
            "InputVolume", self.ui.inputSelector.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            "InputSementation", self.ui.inputSegmentSelector.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            "OutputEnum", self.ui.outputSelector.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            "alignHelperVolume", self.ui.alignHelperSelector.currentNodeID
        )
        # self._parameterNode.SetParameter(
        #     "NumLayers", str(int(self.ui.imageNumLayersSliderWidget.value))
        # )
        self._parameterNode.SetParameter(
            "SavePathSource", str(self.ui.saveField.currentPath)
        )  # not working, TBD
        self._parameterNode.SetParameter(
            "SavePathSegm", str(self.ui.savePathSegm.currentPath)
        )  # not working, TBD
        self._parameterNode.SetParameter(
            "RasCompatible", str(self.ui.checkRasCompatible.checked)
        )
        self._parameterNode.EndModify(wasModified)

        # update GUI to reflect changes made by this method
        self.updateGUIFromParameterNode()

    def getSortingLogic(self):
        """
        Return dictionary of sorting preferences
        depending on the checkbox stats in UI.
        """
        return (
            slogic.sorting_order_ras
            if self.ui.checkRasCompatible.checked
            else slogic.sorting_order_classic
        )

    def onAssessButton(self):
        """
        Run processing when user clicks "Assess" button.
        """
        with slicer.util.tryWithErrorDisplay(
            "Failed to compute results.", waitCursor=True
        ):

            # Compute output
            numSegments = self.logic.processAssess(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.checkVerbose.checked,
            )
            self.ui.assessLabel.text = f"Number of segments: {numSegments}"

    def onErosionButton(self):
        """
        Run processing when user clicks "Binary Erosion" button.
        """
        with slicer.util.tryWithErrorDisplay(
            "Failed to compute results.", waitCursor=True
        ):

            # Compute output
            self.logic.processErosion(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.checkVerbose.checked,
            )

    def onRemoveObButton(self):
        """
        Run processing when user clicks "Remove Small Objects" button.
        """
        with slicer.util.tryWithErrorDisplay(
            "Failed to compute results.", waitCursor=True
        ):

            # Compute output
            self.logic.processRemoveSmallObj(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.morphoSizeSlider.value,
                self.ui.checkVerbose.checked,
            )

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay(
            "Failed to compute results.", waitCursor=True
        ):

            # Compute output
            self.logic.processApply(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.outputSelector.currentNode(),
                self.ui.imageNumLayersSliderWidget.value,
                self.ui.numRowsSlider.value,
                self.ui.mbasesEdit.text,
                self.getSortingLogic(),
                self.ui.checkVerbose.checked,
            )

    def onBreakButton(self):
        """
        Run processing when user clicks "Break" button.
        """
        success = False
        with slicer.util.tryWithErrorDisplay("Failed to break.", waitCursor=True):

            # Compute output
            self.logic.processBreak(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.outputSelector.currentNode(),
                int(self.ui.marginSliderWidget.value),
                self.ui.namePrefixEdit.text,
                self.ui.checkVerbose.checked,
            )
            success = True
        if success:
            slicer.util.infoDisplay(
                "Processing completed successfully.",
                windowTitle="ArrayWranglerModule",
            )

    def onExportButton(self):
        """
        Run processing when user clicks "Export to the local file system" button.
        """
        success = False
        with slicer.util.tryWithErrorDisplay("Failed to export.", waitCursor=True):

            # Compute output
            self.logic.processExport(
                self._parameterNode.GetParameter("Key_local"),
                self.ui.saveField.currentPath,
                self.ui.savePathSegm.currentPath,
                self.ui.checkVerbose.checked,
            )
            success = True
        if success:
            slicer.util.infoDisplay(
                "Processing completed successfully.",
                windowTitle="ArrayWranglerModule",
            )
            self._parameterNode.SetParameter("Key_local", "")
            self.onRefreshLocalButton()
            # self.ui.treeDatasetLocal.clear()

    def onActivateHelperButton(self):
        """
        When (Re)Activate Helper button is clicked
        It creates a helper object with visualized axes
        to help in alignment of initial source, mask
        and segmentation nodes
        """
        logging.debug("(Re)activate helper clicker")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.activateHelper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                self.ui.checkRasCompatible.checked,
            )

    def onFlip0Button(self):
        """
        When the button 'Flip axis Y (1 - Rows)' is clicked
        It flips(reflects) voxels the 0 axis of the volume(s)
        """
        logging.debug("Flip axis Y (1 - Rows) clicked")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            # self.logic.flip0(
            self.logic.array_manipulation_wrapper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                np.flip,
                axis=self.getSortingLogic().get("rows", 0),  # flip along Y axis
            )

    def onFlip1Button(self):
        """
        When the button 'Flip axis X2' is clicked
        It flips(reflects) voxels the 1 axis of the volume(s)
        """
        logging.debug("Flip axis X2 clicked")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.array_manipulation_wrapper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                np.flip,
                axis=self.getSortingLogic().get("columns", 1),  # flip along Y axis
            )

    def onFlip2Button(self):
        """
        When the button 'Flip axis Z3' is clicked
        It flips(reflects) voxels the 2 axis of the volume(s)
        """
        logging.debug("Flip axis Z3 clicked")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.array_manipulation_wrapper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                np.flip,
                axis=self.getSortingLogic().get("height", 2),  # flip along Z axis
            )

    def onSwap01Button(self):
        """
        When the button 'Swap axis 1 and 2' is clicked
        It swaps (transposes) voxels of axes 0 ans 1 axis of the volume(s)
        """
        logging.debug("Swap axis 1 and 2")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.array_manipulation_wrapper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                np.swapaxes,
                axis1=self.getSortingLogic().get("rows", 0),
                axis2=self.getSortingLogic().get("columns", 1),  # swap axes 0 and 1
            )

    def onSwap02Button(self):
        """
        When the button 'Swap axis 1 and 3' is clicked
        It swaps (transposes) voxels of axes 0 ans 2 axis of the volume(s)
        """
        logging.debug("Swap axis 1 and 3")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.array_manipulation_wrapper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                np.swapaxes,
                axis1=self.getSortingLogic().get("rows", 0),
                axis2=self.getSortingLogic().get("height", 2),  # swap axes 0 and 2
            )

    def onSwap12Button(self):
        """
        When the button 'Swap axis 2 and 3' is clicked
        It swaps (transposes) voxels of axes 1 ans 2 axis of the volume(s)
        """
        logging.debug("Swap axis 2 and 3")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.array_manipulation_wrapper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                np.swapaxes,
                axis1=self.getSortingLogic().get("columns", 1),
                axis2=self.getSortingLogic().get("height", 2),  # swap axes 1 and 2
            )

    def onRotZCButton(self):
        """
        When the button 'Rotate axis Z clockwise' is clicked
        It rotates the volume(s) around the Z axis clockwise
        """
        logging.debug("Rotate axis Z clockwise")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.array_manipulation_wrapper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                np.rot90,
                axes=(self.getSortingLogic().get("rotation_z", (1, 0))),
            )

    def onRotZCCButton(self):
        """
        When the button 'Rotate axis Z clockwise' is clicked
        It rotates the volume(s) around the Z axis clockwise
        """
        logging.debug("Rotate axis Z clockwise")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.array_manipulation_wrapper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                np.rot90,
                axes=(self.getSortingLogic().get("rotation_z", (1, 0))[::-1]),
            )

    def onRotXCButton(self):
        """
        When the button 'Rotate axis X clockwise' is clicked
        It rotates the volume(s) around the X axis clockwise
        """
        logging.debug("Rotate axis X clockwise")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.array_manipulation_wrapper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                np.rot90,
                axes=(self.getSortingLogic().get("rotation_x", (2, 0))),
            )

    def onRotXCCButton(self):
        """
        When the button 'Rotate axis X counterclockwise' is clicked
        It rotates the volume(s) around the X axis counterclockwise
        """
        logging.debug("Rotate axis X counterclockwise")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.array_manipulation_wrapper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                np.rot90,
                axes=(self.getSortingLogic().get("rotation_x", (2, 0)))[::-1],
            )

    def onRotYCButton(self):
        """
        When the button 'Rotate axis Y clockwise' is clicked
        It rotates the volume(s) around the Y axis clockwise
        """
        logging.debug("Rotate axis Y clockwise")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.array_manipulation_wrapper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                np.rot90,
                axes=(self.getSortingLogic().get("rotation_y", (1, 2))),
            )

    def onRotYCCButton(self):
        """
        When the button 'Rotate axis Y counterclockwise' is clicked
        It rotates the volume(s) around the Y axis counterclockwise
        """
        logging.debug("Rotate axis Y counterclockwise")
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.array_manipulation_wrapper(
                self.ui.inputSelector.currentNode(),
                self.ui.inputSegmentSelector.currentNode(),
                self.ui.alignHelperSelector.currentNode(),
                np.rot90,
                axes=(self.getSortingLogic().get("rotation_y", (1, 2)))[::-1],
            )

    def onDatasetAvailClicked(self, index):
        """
        Handle click events on the available datasets items.
        When some item is selected it is stored in parameter node
        for subsequent uploading action.
        """

        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            logging.debug(f"Tree item clicked {index = }")
            model = self.ui.treeDatasetLocal.model()
            if model.hasChildren(index):
                key = model.data(index, Qt.DisplayRole)
            else:
                key = model.data(model.parent(index))
            self._parameterNode.SetParameter("Key_local", key)
            logging.debug(f"{key = }")

    def onRefreshLocalButton(self):
        """
        Run processing when user clicks "Refresh" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to process", waitCursor=True):
            self.logic.processRefresh(self.ui.treeDatasetLocal)

    def onEvaluateMaxDimButton(self):
        """
        Run processing when user clicks "Check max size among samples" button.
        """
        success = False
        with slicer.util.tryWithErrorDisplay("Failed to evaluate.", waitCursor=True):

            # Compute output
            max0, max1, max2 = self.logic.processEvaluateMaxDim(
                self._parameterNode.GetParameter("Key_local"),
                self.ui.checkVerbose.checked,
            )
            self.ui.maxSizeEdit.text = f"{max0} {max1} {max2}"

    def onSetNewShapeButton(self):
        """
        Run processing when user clicks "Set new unified shappe" button.
        """
        success = False
        with slicer.util.tryWithErrorDisplay("Failed to evaluate.", waitCursor=True):

            # Compute output
            self.logic.processNewShape(
                self._parameterNode.GetParameter("Key_local"),
                self.ui.maxSizeEdit.text,
                self.ui.checkVerbose.checked,
            )
            # self.ui.maxSizeEdit.text = f"{max0} {max1} {max2}"
            success = True
        if success:
            slicer.util.infoDisplay(
                "Processing completed successfully.",
                windowTitle="ArrayWranglerModule",
            )
            self._parameterNode.SetParameter("Key_local", "")
            self.onRefreshLocalButton()


#
# ArrayWranglerModuleLogic
#
class ArrayWranglerModuleLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    """

    def __init__(self):
        """
        Called when the logic class is instantiated.
        Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        # if not parameterNode.GetParameter("NumLayers"):
        #    parameterNode.SetParaupdateGUIFromParameterNodemeter("NumLayers", "4")

    def processAssess(
        self,
        inputVolume,
        inputMask,
        boolVerbose=False,
    ):
        """
        Run the algorithm to evaluate the number of segments.
        :param inputVolume: source volume to serve as reference
        :param inputMask: mask volume to be used for enumeration
        :param boolVerbose: if True, then print debug information

        """
        if not inputVolume or not inputMask:
            raise ValueError("Input mask or reference input volume is invalid")

        if boolVerbose:
            logging.info("Processing started")

        # label_img = slicer.util.arrayFromVolume(inputVolume).astype(np.uint8)
        # label_img = slicer.util.arrayFromVolume(inputMask).astype(np.uint8)

        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode"
        )
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            inputMask, labelmapVolumeNode, inputVolume
        )
        label_img = slicer.util.arrayFromVolume(labelmapVolumeNode)
        label_img = np.where(label_img > 0, 1, 0)  # binarization

        label_img_enum, _ = ndi.label(label_img)
        enum_labels = np.unique(label_img_enum)[1:]
        if boolVerbose:
            logging.info(f"{label_img.shape = } {label_img.dtype = }")
            logging.info(f"{np.unique(label_img_enum) = }")
            logging.info(f"{enum_labels = }")

        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
        return enum_labels.size

    def processErosion(
        self,
        inputVolume,
        inputMask,
        boolVerbose=False,
    ):
        """
        Run the algorithm to perform binary erosion on the mask.
        It will modify the inputMask in place, and the mask
        will remain binary.
        :param inputVolume: source volume to serve as reference
        :param inputMask: mask volume to be used for enumeration
        :param boolVerbose: if True, then print debug information

        """
        if not inputVolume or not inputMask:
            raise ValueError("Input mask or reference input volume is invalid")

        if boolVerbose:
            logging.info("Processing started")

        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode"
        )
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            inputMask, labelmapVolumeNode, inputVolume
        )
        label_img = slicer.util.arrayFromVolume(labelmapVolumeNode)
        label_img = label_img.astype(bool)  # binarization
        if boolVerbose:
            logging.info(
                f"Before erosion: {label_img.shape=}, {np.count_nonzero(label_img)=}"
            )
        label_img = ski.morphology.binary_erosion(label_img).astype(np.uint8)
        if boolVerbose:
            logging.info(
                f"After erosion {label_img.shape=}, {np.count_nonzero(label_img)=}"
            )
        if np.count_nonzero(label_img) == 0:
            slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
            slicer.mrmlScene.RemoveNode(inputMask)
            return

        slicer.util.updateVolumeFromArray(labelmapVolumeNode, label_img)

        # slicer.util.updateVolumeFromArray(outputVolume, label_img_enum_copy)
        colorTableNode = setColorTable([0, 1])
        labelmapVolumeNode.GetDisplayNode().SetAndObserveColorNodeID(
            colorTableNode.GetID()
        )
        inputMask.GetSegmentation().RemoveAllSegments()  # in case we reuse
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelmapVolumeNode, inputMask
        )
        inputMask.CreateClosedSurfaceRepresentation()
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

    def processRemoveSmallObj(
        self,
        inputVolume,
        inputMask,
        obSize,
        boolVerbose=False,
    ):
        """
        Run the algorithm to perform binary erosion on the mask.
        It will modify the inputMask in place, and the mask
        will remain binary.
        :param inputVolume: source volume to serve as reference
        :param inputMask: mask volume to be used for enumeration
        :param obSize: size threshold (in voxels) below which objects will be removed
        :param boolVerbose: if True, then print debug information

        """
        if not inputVolume or not inputMask:
            raise ValueError("Input mask or reference input volume is invalid")

        if boolVerbose:
            logging.info("Processing started")

        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode"
        )
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            inputMask, labelmapVolumeNode, inputVolume
        )
        label_img = slicer.util.arrayFromVolume(labelmapVolumeNode)
        label_img = label_img.astype(bool)  # binarization
        if boolVerbose:
            logging.info(f"Before removal: {label_img.shape=} {label_img.dtype=}")
        label_img = ski.morphology.remove_small_objects(label_img, obSize).astype(
            np.uint8
        )
        if boolVerbose:
            logging.info(f"After removal {label_img.shape=} {label_img.dtype=}")

        slicer.util.updateVolumeFromArray(labelmapVolumeNode, label_img)

        # slicer.util.updateVolumeFromArray(outputVolume, label_img_enum_copy)
        colorTableNode = setColorTable([0, 1])
        labelmapVolumeNode.GetDisplayNode().SetAndObserveColorNodeID(
            colorTableNode.GetID()
        )
        inputMask.GetSegmentation().RemoveAllSegments()  # in case we reuse
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelmapVolumeNode, inputMask
        )
        inputMask.CreateClosedSurfaceRepresentation()
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

    def processApply(
        self,
        inputVolume,
        inputMask,
        outputVolume,
        numLayers,
        numRows=3,  # TODO make it variable like mbases list
        mbases="",
        sorting_order=slogic.sorting_order_classic,
        boolVerbose=False,
    ):
        """
        Run the enumeration algorithm.

        :param inputVolume: volume to be thresholded
        :param inputMask: mask volume to be used for enumeration
        :param outputVolume: enumerated result
        :param numLayers: number of layers in stack to be used for clustering
        by height
        :param numRows: number of rows on a layer/plate
        :param mbases: mapping bases for enumeration, e.g. "100,200,300,400"
        :param boolVerbose: if True, then print debug information

        """
        if not inputVolume or not outputVolume or not inputMask:
            raise ValueError("Inputs or outputs are invalid")

        # TODO : comment after debugging
        import importlib

        importlib.reload(slogic)
        import time

        numLayers = int(numLayers)
        numRows = int(numRows)
        startTime = time.time()
        if boolVerbose:
            logging.info("Processing started")
            logging.info("Updating output volume")
            logging.info(f"{numLayers = }")
            logging.info(f"{numLayers = }")

        # label_img = slicer.util.arrayFromVolume(inputVolume).astype(np.uint8)
        # label_img = slicer.util.arrayFromVolume(inputMask).astype(np.uint8)

        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode"
        )
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            inputMask, labelmapVolumeNode, inputVolume
        )
        label_img = slicer.util.arrayFromVolume(labelmapVolumeNode)
        label_img = np.where(label_img > 0, 1, 0)  # binarization

        # TODO: remove this workaround when alinement feature is done
        # label_img = np.swapaxes(label_img, 0, 2)  # temporary fix
        label_img_enum, _ = ndi.label(label_img)
        enum_labels = np.unique(label_img_enum)[1:]
        if boolVerbose:
            logging.info(f"{label_img.shape = } {label_img.dtype = }")
            logging.info(f"{np.unique(label_img_enum) = }")
            logging.info(f"{enum_labels = }")

        # slicer.util.updateVolumeFromArray(outputVolume, input_nparray)
        centroids = np.round(
            ndi.center_of_mass(label_img_enum, label_img_enum, enum_labels)
        ).astype(np.int32)

        # 4 numLayers
        sorted_level_labels = slogic.cluster_zcoord(
            cpoints=centroids,
            num_zclusters=numLayers,
            sorder=sorting_order,
            debug=boolVerbose,
        )
        if boolVerbose:
            logging.info(f"{centroids.shape = }")
            logging.info(f"{sorted_level_labels = }")

        if len(mbases) > 0:
            if "," in mbases:
                mapping_bases = [int(mbase) for mbase in mbases.split(",")]
            elif "-" in mbases:
                mapping_bases = [int(mbase) for mbase in mbases.split("-")]
            elif " " in mbases:
                mapping_bases = [int(mbase) for mbase in mbases.split()]
            else:
                logging.error("Wrong format of mapping bases. Canceling operation.")
                return
            assert (
                len(mapping_bases) == numLayers
            ), "Number of mapping bases should be equal to number of layers."
        else:  # mapping_bases = [100, 200, 300, 400] #
            spacer = 100  # the 'space' between level numbers, should be made adjustable
            mapping_bases = np.arange(spacer, (numLayers + 1) * spacer, spacer)
        if boolVerbose:
            logging.info(f"{mapping_bases = }")

        final_remap = slogic.full_remap(
            level_bases=mapping_bases,
            center_points=centroids,
            levelwise_labels=sorted_level_labels,
            init_enum=enum_labels,
            rows_onlevel=numRows,  # TODO: make it variable in a list
            sorting_scheme=sorting_order,
            debug=boolVerbose,
        )
        final_labels = sorted(set(final_remap.values()))
        if boolVerbose:
            logging.info(f"{final_remap = }")

        label_img_enum_copy = label_img_enum.copy()

        # restore the volume orientation from the workarond before
        # TODO: remove this workaround when alinement feature is done
        # label_img_enum_copy = np.swapaxes(label_img_enum_copy, 0, 2)
        label_img_enum_copy = slogic.perform_remap(
            remapping_dict=final_remap, enum_img=label_img_enum_copy
        )

        # logging.info(f'Exporting labelmapVolumeNode level-wise enumeration')

        """
        # These steps would export the label map with original enumaration
        # (having levels starting 100, 200 etc.)
        slicer.util.updateVolumeFromArray(labelmapVolumeNode, label_img_enum_copy)
        slicer.util.exportNode(labelmapVolumeNode,
        "path/labelmapVolumeNode_level-wise.nii")
        """
        # self.temp_enum_array = label_img_enum_copy.copy()

        # Bringing enumaration to consequtive format.
        # Converions between label map and Segmentation node working properly
        # with consequtive labels only
        # Color Table doesn't accept large numbers like 100+, 200+ either
        label_img_enum_copy = slogic.make_consequtive_labels(
            enum_img=label_img_enum_copy, sparse_labels=final_labels
        )
        if boolVerbose:
            logging.info(
                f"{label_img_enum_copy.shape = } {label_img_enum_copy.dtype = }"
            )
        slicer.util.updateVolumeFromArray(labelmapVolumeNode, label_img_enum_copy)

        # slicer.util.updateVolumeFromArray(outputVolume, label_img_enum_copy)
        colorTableNode = setColorTable(final_labels)
        labelmapVolumeNode.GetDisplayNode().SetAndObserveColorNodeID(
            colorTableNode.GetID()
        )
        outputVolume.GetSegmentation().RemoveAllSegments()  # in case we reuse
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelmapVolumeNode, outputVolume
        )
        outputVolume.CreateClosedSurfaceRepresentation()
        stopTime = time.time()
        if boolVerbose:
            logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

        # logging.info(f'Exporting labelmapVolumeNode')
        # slicer.util.exportNode(labelmapVolumeNode,
        # "X:\Yaroslav\SlicerExtensions\labelmapVolumeNode.nii.gz")

    def processBreak(
        self,
        inputNode,
        maskNode,
        enumeratedNode,
        span,
        # boolPreserve,
        # savePathSource,
        # savePathSegm,
        namePrefix,
        boolVerbose,
    ):
        """
        Run the processing algorithm to break the source volume into smaller cubicles
        based on the mask volume.
        Parameters:
        inputNode (vtkMRMLScalarVolumeNode): Node with the source volume to be broken.
        maskNode (vtkMRMLScalarVolumeNode): Node with the mask volume for breaking.
        enumeratedNode (vtkMRMLScalarVolumeNode): Node with the enumerated volume.
        Returns:
        None
        The function performs the following steps:
        1. Converts the input volume to a numpy array.
        2. Uses a temporary enumerated array for processing.
        3. Finds objects in the enumerated array.
        4. Expands the dimensions of the found objects by a specified number of voxels.
        5. Breaks the expanded objects into smaller cubicles.
        6. Creates new volume nodes for each cubicle and adds them to the scene.

        """
        # assert len(savePathSource) > 0 , "No output paths selected"
        boolPreserve = True  # saving objects in the Slicer scene

        source_img = slicer.util.arrayFromVolume(inputNode).astype(np.int16)
        # enum_img = slicer.util.arrayFromVolume(inputMask).astype(np.uint8)

        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode"
        )

        # This node will serve to transfer ColorTable from segmentation
        # node to separated pieces
        labelmapSegNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode"
        )

        # You can use this, when inputNode is not available
        # but make sure slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY is there
        # otherwise the image will be cropped
        # slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
        # segmentationNode, labelmapVolumeNode,
        # slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            enumeratedNode, labelmapVolumeNode, inputNode
        )
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            maskNode, labelmapSegNode, inputNode
        )
        enum_array = slicer.util.arrayFromVolume(labelmapVolumeNode)
        segm_img = slicer.util.arrayFromVolume(labelmapSegNode)
        node_segmentation = enumeratedNode.GetSegmentation()
        seg_ids = list(node_segmentation.GetSegmentIDs())
        seg_map = {}
        for seg_id in seg_ids:
            segment = node_segmentation.GetSegment(seg_id)
            seg_map[segment.GetLabelValue()] = segment.GetName()
        assert set(seg_map.keys()) == set(
            np.unique(enum_array)[1:]
        ), "Segmentation labels and enumerated labels are not matching"

        obfound = ndi.find_objects(enum_array)  # labels_dbscan  labels_watershed

        # span = 5
        ymax, xmax, zmax = enum_array.shape  #

        ob_expanded = slogic.expand_object_dims(
            obj_list=obfound, span=span, ymax0=ymax, xmax1=xmax, zmax2=zmax
        )

        if boolVerbose:
            logging.info(f"{len(obfound) = }")
            logging.info(f"{len(ob_expanded) = }")

        if boolVerbose:
            logging.info("Processing obcubes_mask")
        # so far unused
        obcubes_mask, shapes = slogic.break_cubicles(ob_expanded, enum_array)

        if boolVerbose:
            logging.info("Processing obcubes_source")
        obcubes_source, _ = slogic.break_cubicles(ob_expanded, source_img)

        if boolVerbose:
            logging.info("Processing obcubes_segm")
        obcubes_segm, _ = slogic.break_cubicles(ob_expanded, segm_img)

        # cleaning segmented cubicles by restricting the segmentation to
        # enumerated mask (prevents the snapping of adjacent seeds)
        obcubes_segm_clean = []
        unique_labels = np.unique(enum_array)[1:]
        for i, obcube_segm in enumerate(obcubes_segm):
            lb_index = unique_labels[i]
            lb_mask = obcubes_mask[i]
            obcube_segm_new = np.where(lb_mask == lb_index, obcube_segm, 0)
            obcubes_segm_clean.append(obcube_segm_new)

        if boolVerbose:
            logging.info(f"{len(obcubes_mask) = }")
            logging.info(f"{len(obcubes_source) = }")
            logging.info(f"{len(obcubes_segm_clean) = }")

        # temp_path = r"\\filer-5\user\plutenko\projects\gitlab\playground"
        # slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")

        # color_table_seg = labelmapSegNode.GetDisplayNode().GetColorNode()
        # original_color_table_id = color_table_seg.GetID()
        original_color_table_id = labelmapSegNode.GetDisplayNode().GetColorNodeID()

        # print(f"{original_color_table_id = }")
        # TBD processing for segmentation node

        # Getting Subject Hierarchy node to arrange newly created nodes
        # under a folder node
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        sceneItemID = shNode.GetSceneItemID()
        folderName = inputNode.GetName()
        if len(namePrefix) > 0:
            folderName = namePrefix + "_" + folderName
        # checking if the folder with given name already exist
        if shNode.GetItemByName(folderName) != 0:
            folderName = f"{folderName}_{str(np.random.randint(np.iinfo(np.int16).max))}"  # 32767

        dataFolderNodeID = shNode.CreateFolderItem(sceneItemID, folderName)
        shNode.SetItemAttribute(
            dataFolderNodeID, FOLDER_ATTRIBUTE, FOLDER_ATTRIBUTE_VALUE
        )
        for lb, ob_source in enumerate(obcubes_source):
            ob_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
            # deprecated version
            # node_name = seg_map.get(lb + 1, f"Unknown_Segment_{lb+1}").replace(
            #     "Segment", namePrefix
            # )
            node_name = seg_map.get(lb + 1, f"Unknown_Segment_{lb+1}").replace(
                "Segment_", ""  # only label names to be left
            )
            ob_node.SetName(node_name)
            slicer.util.updateVolumeFromArray(ob_node, ob_source)
            slicer.mrmlScene.AddNode(ob_node)
            # getting id of newly created node in the hierarchy space
            hierarchyItemID = shNode.GetItemChildWithName(sceneItemID, node_name)
            # and putting it under the folder item
            shNode.SetItemParent(hierarchyItemID, dataFolderNodeID)
            # ob_node.UnRegister(None)
            # if len(savePathSource) > 0:
            #     fname = node_name + "_0000.nii.gz"
            #     slicer.util.exportNode(ob_node, os.path.join(savePathSource, fname))

            # obseg_node = slicer.vtkMRMLSegmentationNode()
            # slicer.mrmlScene.AddNode(obseg_node)
            obseg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            obseg_node.CreateDefaultDisplayNodes()  # only needed for display
            obseg_node.SetReferenceImageGeometryParameterFromVolumeNode(ob_node)
            obseg_lbmapnode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode"
            )
            obseg_node.SetName(node_name)
            obseg_lbmapnode.CreateDefaultDisplayNodes()
            slicer.util.updateVolumeFromArray(obseg_lbmapnode, obcubes_segm_clean[lb])
            obseg_lbmapnode.GetDisplayNode().SetAndObserveColorNodeID(
                original_color_table_id
            )
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                obseg_lbmapnode, obseg_node
            )
            ob_node.UnRegister(None)
            # getting id of newly created node in the hierarchy space under the root
            # the node name is the same as for source volume, but it has been
            # moved under the folder already
            hierarchyItemID = shNode.GetItemChildWithName(sceneItemID, node_name)
            shNode.SetItemParent(hierarchyItemID, dataFolderNodeID)
            # if len(savePathSegm) > 0:
            #     fsegname = node_name + ".nii.gz"
            #     slicer.util.exportNode(
            #         obseg_lbmapnode, os.path.join(savePathSegm, fsegname)
            #     )
            if not boolPreserve:
                slicer.mrmlScene.RemoveNode(ob_node)
                slicer.mrmlScene.RemoveNode(obseg_node)
            slicer.mrmlScene.RemoveNode(obseg_lbmapnode)
        slicer.mrmlScene.RemoveNode(labelmapSegNode)
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

    def activateHelper(
        self,
        inputNode,
        maskNode,
        helperNode,
        rasCompatible=False,
    ):
        """
        Run the processing to activate a helper object with visualized axes
        based on the input (and mask) volume(s).
        Parameters:
        inputNode (vtkMRMLScalarVolumeNode): Node with the source volume to be broken.
        maskNode (vtkMRMLSegmentationNode): Node with the mask volume for breaking.
        helperNode (vtkMRMLSegmentationNode): Node with the helper volume to be created.
        rasCompatible (bool): If True, the helper axes will be drawn in RAS orientation.
        Returns:
        None
        The function performs the following steps:
        1. Converts the input volume to a numpy array.
        2. Creates a new empty numpy with the same shape as the input volume.
        3. Draws axes in a new volume with different lengh to differentiate.
        4. Converts/assigns the new volume with axes to a helper segmentation node.

        """
        # TODO: comment after debugging
        import importlib

        importlib.reload(slogic)
        if not inputNode or not helperNode:
            raise ValueError("Input or Helper object(node) is invalid")

        # Cleaning the helper node (everything will be recreated anyway)
        # if helperNode.GetNumberOfSegments() > 0:
        helperNode.GetSegmentation().RemoveAllSegments()

        # Create a new empty numpy array with the same shape as the input volume
        helper_array = np.zeros_like(slicer.util.arrayFromVolume(inputNode))

        # TODO: remove this workaround when alinement feature is done
        # helper_array = np.swapaxes(helper_array, 0, 2)  # temporary fix

        # Get the dimensions of the input volume
        z_dim, y_dim, x_dim = helper_array.shape

        # Draw axes in the helper array
        # The axes are drawn with different lengths to differentiate them
        if rasCompatible:
            helper_array = slogic.draw_axes_ras(
                vol_image=helper_array, offset=5, line_width=5
            )
        else:
            helper_array = slogic.draw_axes(
                vol_image=helper_array, offset=5, line_width=5
            )

        # Temporary label map node before converting to segmentation node
        labelmapHelperNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode"
        )

        colorHelperTable = setHelperTable()
        labelmapHelperNode.CreateDefaultDisplayNodes()
        labelmapHelperNode.GetDisplayNode().SetAndObserveColorNodeID(
            colorHelperTable.GetID()
        )

        slicer.util.updateVolumeFromArray(labelmapHelperNode, helper_array)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelmapHelperNode, helperNode
        )
        helperNode.SetName("Helper Segmentation (Axes)")
        # helperNode.GetDisplayNode().SetVisibility(True) # already by default
        helperNode.CreateClosedSurfaceRepresentation()
        slicer.mrmlScene.RemoveNode(labelmapHelperNode)

        # debugging
        # labelmapVolumeNode_debug = slicer.mrmlScene.AddNewNodeByClass(
        #     "vtkMRMLLabelMapVolumeNode"
        # )
        # slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
        #     helperNode,
        #     labelmapVolumeNode_debug,
        #     slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
        # debug_volume = slicer.util.arrayFromVolume(labelmapVolumeNode_debug)
        # print(f"{debug_volume.shape = } {debug_volume.dtype = }")
        # slicer.mrmlScene.RemoveNode(labelmapVolumeNode_debug)

    def array_manipulation_wrapper(
        self, inputNode, maskNode, helperNode, func, *args, **kwargs
    ):
        """
        This function is used to manipulate the voxels in the input volume(s)
        Parameters:
        inputNode (vtkMRMLScalarVolumeNode): Node with the source volume to be flipped.
        maskNode (vtkMRMLSegmentationNode): Node with the mask to be flipped.
        helperNode (vtkMRMLSegmentationNode): Node with the helper volume
        to be recreated.
        func (function): Numpy operation Function to be applied to the input
        and mask arrays.
        *args: Additional arguments to be passed to the function.
        **kwargs: Additional keyword arguments to be passed to the function.
        Returns:
        None
        """
        logging.debug(f"{func.__name__} called with {args = } and {kwargs = }")
        src_volume = slicer.util.arrayFromVolume(inputNode)

        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode"
        )

        # slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
        #     maskNode,
        #     labelmapVolumeNode,
        #     slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
        # this is preferred to ExportVisibleSegmentsToLabelmapNode
        # as it preserves the volume extents matching the inputNode
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            maskNode, labelmapVolumeNode, inputNode
        )
        mask_volume = slicer.util.arrayFromVolume(labelmapVolumeNode)

        # TODO: remove this workaround when alinement feature is done
        # src_volume = np.swapaxes(src_volume, 0, 2)  # temporary fix
        # mask_volume = np.swapaxes(mask_volume, 0, 2)  # temporary fix
        logging.debug("Before manipulation:")
        logging.debug(f"{src_volume.shape = } {src_volume.dtype = }")
        logging.debug(f"{mask_volume.shape = } {mask_volume.dtype = }")
        # src_volume = np.flip(src_volume, axis=0)  # flip along Y axis
        # mask_volume = np.flip(mask_volume, axis=0)
        src_volume = func(
            src_volume, *args, **kwargs
        )  # apply the function to the source volume
        mask_volume = func(
            mask_volume, *args, **kwargs
        )  # apply the function to the mask volume
        logging.debug("After manipulation:")
        logging.debug(f"{src_volume.shape = } {src_volume.dtype = }")
        logging.debug(f"{mask_volume.shape = } {mask_volume.dtype = }")

        slicer.util.updateVolumeFromArray(inputNode, src_volume)

        maskNode.GetSegmentation().RemoveAllSegments()
        slicer.util.updateVolumeFromArray(labelmapVolumeNode, mask_volume)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelmapVolumeNode, maskNode
        )
        maskNode.CreateClosedSurfaceRepresentation()
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
        # print(f"{self.getParameterNode().GetParameter('RasCompatible')=}")
        self.activateHelper(
            inputNode=inputNode,
            maskNode=maskNode,
            helperNode=helperNode,
            rasCompatible=self.getParameterNode().GetParameter("RasCompatible")
            == "True",
        )

    def processRefresh(self, localTreeWidget):
        """
        processRefresh: Refresh the local datasets tree widget.
        localTreeWidget (QTreeWidget): The tree widget to populate with local datasets.
        """
        populateLocalDatasets(localTreeWidget)

    def processExport(
        self,
        datasetName,
        savePathSource,
        savePathSegm,
        boolVerbose,
    ):
        """
        Run the algorithm to export the source and/or label volumes into a local
        file system, the place specified in the respective UI fields.
        based on the mask volume.
        Parameters:
        datasetName (string): Parent folder name containing samples
        savePathSource (string): Node with the source volume to be broken.
        savePathSegm (string): Node with the mask volume for breaking.
        enumeratedNode (vtkMRMLScalarVolumeNode): Node with the enumerated volume.
        Returns:
        None

        """
        logging.debug(f"{datasetName = }")
        logging.debug(f"{savePathSource = }")
        logging.debug(f"{savePathSegm = }")
        assert len(datasetName) > 0, "Dataset name is not specified!"

        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        if len(savePathSource) > 0:
            volNodesDict = extractNodes(
                shNd=shNode, key_name=datasetName, nodeClass="vtkMRMLScalarVolumeNode"
            )
            for i, (node_name, volumeNode) in enumerate(volNodesDict.items()):
                fname = node_name + "_0000.nii.gz"
                slicer.util.exportNode(volumeNode, os.path.join(savePathSource, fname))

        if len(savePathSegm) > 0:
            segNodesDict = extractNodes(
                shNd=shNode, key_name=datasetName, nodeClass="vtkMRMLSegmentationNode"
            )
            for i, (node_name, segNode) in enumerate(segNodesDict.items()):
                fname = node_name + ".nii.gz"
                labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLLabelMapVolumeNode"
                )

                slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                    segNode,
                    labelmapVolumeNode,
                    slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY,
                )

                slicer.util.exportNode(
                    labelmapVolumeNode, os.path.join(savePathSegm, fname)
                )
                slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

    def processEvaluateMaxDim(
        self,
        datasetName,
        boolVerbose=False,
    ):
        """
        Evaluating the maximum span that samples occupy
        to determine the subsequent expansion(padding) of images

        Parameters:
        datasetName (string): Parent folder name containing samples
        Returns:
        tuple of max dimensions among samples

        """
        logging.debug(f"{datasetName = }")
        assert len(datasetName) > 0, "Dataset name is not specified!"

        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        shapes = []
        volNodesDict = extractNodes(
            shNd=shNode, key_name=datasetName, nodeClass="vtkMRMLScalarVolumeNode"
        )
        # print(volNodesDict)
        for i, (node_name, volumeNode) in enumerate(volNodesDict.items()):
            shapes.append(slicer.util.arrayFromVolume(volumeNode).shape)
        shape_np = np.array(shapes)
        logging.debug(f"{shape_np.shape = }")
        logging.debug(
            f"{shape_np.max() = }, {np.max(shape_np[:, 0]) = }, {np.max(shape_np[:, 1]) = }, {np.max(shape_np[:, 2]) = }"
        )
        return np.max(shape_np[:, 0]), np.max(shape_np[:, 1]), np.max(shape_np[:, 2])

    def processNewShape(
        self,
        datasetName,
        dimensions,
        boolVerbose=False,
    ):
        """
        Evaluating the maximum span that samples occupy
        to determine the subsequent expansion(padding) of images

        Parameters:
        datasetName (string): Parent folder name containing samples
        dimensions (string): three values with delimiter setting new shape for samples
        Returns:
        None

        """
        logging.debug(f"{datasetName = }")
        assert len(datasetName) > 0, "Dataset name is not specified!"
        assert len(dimensions) > 0, "Dimensions are not specified"

        if "," in dimensions:
            dims = [int(dm) for dm in dimensions.split(",")]
        elif "-" in dimensions:
            dims = [int(dm) for dm in dimensions.split("-")]
        elif " " in dimensions:
            dims = [int(dm) for dm in dimensions.split()]
        else:
            logging.error("Wrong format of mapping bases. Canceling operation.")
            return
        logging.debug(f"{dims = }")
        assert len(dims) == 3, "Wrong number of dimensions"

        rmax0, rmax1, rmax2 = self.processEvaluateMaxDim(datasetName)
        assert (
            rmax0 <= dims[0] and rmax1 <= dims[1] and rmax2 <= dims[2]
        ), f"Some specified dimension(s) is smaller than maximum dimension in the dataset {rmax0, rmax1, rmax2 = }"

        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        volNodesDict = extractNodes(
            shNd=shNode, key_name=datasetName, nodeClass="vtkMRMLScalarVolumeNode"
        )
        segNodesDict = extractNodes(
            shNd=shNode, key_name=datasetName, nodeClass="vtkMRMLSegmentationNode"
        )
        logging.debug(f"{len(segNodesDict) = }")

        sceneItemID = shNode.GetSceneItemID()
        folderName = f"{datasetName}_{dims[0]}x{dims[1]}x{dims[2]}"
        # checking if the folder with given name already exist
        if shNode.GetItemByName(folderName) != 0:
            folderName = f"{folderName}_{str(np.random.randint(np.iinfo(np.int16).max))}"  # 32767
        dataFolderNodeID = shNode.CreateFolderItem(sceneItemID, folderName)
        shNode.SetItemAttribute(
            dataFolderNodeID, FOLDER_ATTRIBUTE, FOLDER_ATTRIBUTE_VALUE
        )

        # import importlib
        # importlib.reload(slogic)
        labelmapSegNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode"
        )
        for i, (node_name, volumeNode) in enumerate(volNodesDict.items()):
            source_img = slicer.util.arrayFromVolume(volumeNode).astype(np.int16)
            logging.debug(f"{source_img.shape = }")
            source_img = slogic.pad_volume(source_img, dims)
            logging.debug(f"{source_img.shape = }")
            # ob_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")
            itemIDToClone = shNode.GetItemByDataNode(volumeNode)
            clonedItemID = (
                slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(
                    shNode, itemIDToClone
                )
            )
            ob_node = shNode.GetItemDataNode(clonedItemID)
            shNode.SetItemParent(clonedItemID, dataFolderNodeID)
            ob_node.SetName(node_name)
            slicer.util.updateVolumeFromArray(ob_node, source_img)

            segNode = segNodesDict.get(node_name, None)
            if segNode is not None:
                slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                    segNode,
                    labelmapSegNode,
                    slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY,
                )
                label_img = slicer.util.arrayFromVolume(labelmapSegNode).astype(
                    np.int16
                )
                logging.debug(f"{label_img.shape = }")
                label_img = slogic.pad_volume(label_img, dims)
                logging.debug(f"{label_img.shape = }")
                # obseg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                itemIDToClone = shNode.GetItemByDataNode(segNode)
                clonedItemID = (
                    slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(
                        shNode, itemIDToClone
                    )
                )
                obseg_node = shNode.GetItemDataNode(clonedItemID)

                obseg_node.SetReferenceImageGeometryParameterFromVolumeNode(ob_node)
                clonedItemID = shNode.GetItemByDataNode(obseg_node)
                shNode.SetItemParent(clonedItemID, dataFolderNodeID)
                obseg_node.SetName(node_name)
                obseg_node.GetSegmentation().RemoveAllSegments()
                slicer.util.updateVolumeFromArray(labelmapSegNode, label_img)
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                    labelmapSegNode, obseg_node
                )
        slicer.mrmlScene.RemoveNode(labelmapSegNode)


#
# ArrayWranglerModuleTest
#
class ArrayWranglerModuleTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state
        - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_ArrayWranglerModule1()

    def test_ArrayWranglerModule1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        self.delayDisplay("Test passed")


def setColorTable(mapped_labels):
    # assumed that mapped labels should be sorted in the order
    # of the consequtive labels
    # otherwise the color table will be messed with label_map
    # we can pass the sorted labels to the function,
    # and/or check the consistency of labels/segments after returning the function
    # UPD: the checkup actually performed in processBreak
    colorTableNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLColorTableNode")
    colorTableNode.SetTypeToUser()
    colorTableNode.HideFromEditorsOff()  # make selectable in the GUI
    slicer.mrmlScene.AddNode(colorTableNode)
    colorTableNode.UnRegister(None)
    largestLabelValue = len(mapped_labels)
    colorTableNode.SetNumberOfColors(largestLabelValue + 1)
    # prevent automatic color name generation
    colorTableNode.SetNamesInitialised(True)

    g = random.uniform(0.0, 1.0)
    b = random.uniform(0.0, 1.0)
    r = random.uniform(0.0, 1.0)
    a = 1.0

    coff = 1 / len(mapped_labels)  # for making progressive coloring
    for i, labelValue in enumerate(mapped_labels):
        r = coff * (i + 1)
        segmentName = f"Segment_{labelValue}_{i+1}"
        # Can use success variable instead of _ to check if performed ok
        _ = colorTableNode.SetColor(i + 1, segmentName, r, g, b, a)

    return colorTableNode


def setHelperTable():
    # setting up a color table for the helper object
    # using three colors/names for the axes
    mapped_labels = {
        1: "Y Axis (Rows)",
        2: "X Axis (Columns, or seeds in a row)",
        3: "Z Axis/Height (Layers/plates in a stack)",
    }
    colorTableNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLColorTableNode")
    colorTableNode.SetTypeToUser()
    colorTableNode.HideFromEditorsOff()  # make selectable in the GUI
    slicer.mrmlScene.AddNode(colorTableNode)
    colorTableNode.UnRegister(None)
    largestLabelValue = len(mapped_labels)
    colorTableNode.SetNumberOfColors(largestLabelValue + 1)
    # prevent automatic color name generation
    colorTableNode.SetNamesInitialised(True)

    g = random.uniform(0.0, 1.0)
    b = random.uniform(0.0, 1.0)
    r = random.uniform(0.0, 1.0)
    a = 1.0

    coff = 1 / len(mapped_labels)  # for making progressive coloring
    for labelValue, name_template in mapped_labels.items():
        r = coff * (labelValue)
        segmentName = f"{name_template}_{labelValue}"
        # Can use success variable instead of _ to check if performed ok
        _ = colorTableNode.SetColor(labelValue, segmentName, r, g, b, a)

    return colorTableNode


def populateLocalDatasets(tree):
    """
    Populate the QTreeWidget with local datasets.
    """

    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    sceneItemID = shNode.GetSceneItemID()
    children = vtk.vtkIdList()  # empty list
    shNode.GetItemChildren(sceneItemID, children)  # populated list
    data = {}
    for id in range(children.GetNumberOfIds()):
        itemID = children.GetId(id)
        itemName = shNode.GetItemName(itemID)
        if shNode.GetItemAttribute(itemID, FOLDER_ATTRIBUTE) == FOLDER_ATTRIBUTE_VALUE:
            subchildren = vtk.vtkIdList()  # empty list
            shNode.GetItemChildren(itemID, subchildren)
            for subid in range(subchildren.GetNumberOfIds()):
                subitemID = subchildren.GetId(subid)
                subitemName = shNode.GetItemName(subitemID)
                data.setdefault(itemName, []).append(subitemName)
    tree.clear()
    items = []
    for key, value in data.items():
        item = QTreeWidgetItem([key])
        for child_name in value:
            child = QTreeWidgetItem([child_name])
            item.addChild(child)
        items.append(item)

    tree.insertTopLevelItems(0, items)


def criteria_node_in_sh(node, shNd, keyname):
    shItemID = shNd.GetItemByDataNode(node)
    parentItem = shNd.GetItemParent(shItemID)
    return (
        shNd.GetItemName(parentItem) == keyname
        and shNd.GetItemAttribute(parentItem, FOLDER_ATTRIBUTE)
        == FOLDER_ATTRIBUTE_VALUE
    )


def extractNodes(shNd, key_name, nodeClass="vtkMRMLScalarVolumeNode"):
    nodes = slicer.util.getNodesByClass(nodeClass)
    nodes = sorted(
        # filter(criteria_node_in_sh, nodes),
        [nd for nd in nodes if criteria_node_in_sh(nd, shNd, key_name)],
        key=lambda node: node.GetName(),
    )
    nodes_dict = {node.GetName(): node for node in nodes}
    return nodes_dict
