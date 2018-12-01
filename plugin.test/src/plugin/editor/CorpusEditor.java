package plugin.editor;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.ResourceBundle;

import org.eclipse.jface.action.IAction;
import org.eclipse.jface.action.IMenuManager;
import org.eclipse.jface.text.BadLocationException;
import org.eclipse.jface.text.IDocument;
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
	
	private class StripNETagAction extends TextEditorAction {
		
		protected StripNETagAction(ResourceBundle bundle, String prefix, ITextEditor editor) {
			super(bundle, prefix, editor);
		}
		
		@Override
		public void run() {
			ITextEditor editor= getTextEditor();
			IDocument document = editor.getDocumentProvider().getDocument(editor.getEditorInput());
			BufferedReader br = new BufferedReader(new StringReader(document.get()));
			StringBuffer sb = new StringBuffer();
			try {
				String line = null;
				while ((line = br.readLine()) != null) {
					String[] splits = line.split("\t");
					if (splits.length < 2) {
						continue;
					}
					splits[1] = splits[1].replaceAll("<([^:]*):[^>]*>", "$1");
					sb.append(String.join("\t", splits) + "\n");
				}
				br.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			try {
				document.replace(0, document.getLength(), sb.substring(0, sb.length()-1).toString());
			} catch (BadLocationException e) {
				e.printStackTrace();
			}
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
		
		a= new StripNETagAction(CorpusEditorMessages.getResourceBundle(), "StripNETagAction.", this);
		setAction("StripNETagAction", a); //$NON-NLS-1$
	}
	
	@Override
	protected void editorContextMenuAboutToShow(IMenuManager menu) {
		super.editorContextMenuAboutToShow(menu);
		addAction(menu, "SelectedTextAction");
		addAction(menu, "StripNETagAction");
	}
}
