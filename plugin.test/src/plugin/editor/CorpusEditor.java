package plugin.editor;

import java.util.ResourceBundle;

import org.eclipse.jface.action.IAction;
import org.eclipse.jface.action.IMenuManager;
import org.eclipse.jface.viewers.ISelection;
import org.eclipse.ui.editors.text.TextEditor;
import org.eclipse.ui.texteditor.ITextEditor;
import org.eclipse.ui.texteditor.TextEditorAction;

public class CorpusEditor extends TextEditor {
	
	private class SelectedTextAction extends TextEditorAction {

		protected SelectedTextAction(ResourceBundle bundle, String prefix, ITextEditor editor) {
			super(bundle, prefix, editor);
		}
		
		@Override
		public void run() {
			ITextEditor editor= getTextEditor();
			ISelection selection= editor.getSelectionProvider().getSelection();
			System.out.println("selection: " + selection.toString());
		}
		
	}

	public CorpusEditor() {
		// TODO Auto-generated constructor stub
	}
	
	@Override
	protected void createActions() {
		super.createActions();

		IAction a= new SelectedTextAction(CorpusEditorMessages.getResourceBundle(), "SelectedTextAction.", this); //$NON-NLS-1$
		setAction("SelectedTextAction", a); //$NON-NLS-1$
	}
	
	@Override
	protected void editorContextMenuAboutToShow(IMenuManager menu) {
		super.editorContextMenuAboutToShow(menu);
		addAction(menu, "SelectedTextAction");
	}
}
