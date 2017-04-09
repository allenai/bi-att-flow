(function() {

	var PNUM = 49
        var paragraph_id = 0;
	var titles;
	var contextss;
        var context_questions;

	window.onload = function(){
		displayQuestion();
		load();
	}

	function clearField(){
		var qadiv = document.getElementById("qa");
		qadiv.innerHTML = "";
		displayQuestion();
	}

	function load(){
		sendAjax("/select", {}, handleParagraph);
	}
	function loadDropdown(){
		var dropdown = document.getElementById("selectArticle");
		for(var i=0; i<PNUM; i++){
			var opt = document.createElement("option");
			opt.value = parseInt(i);
			opt.innerHTML = titles[i];
			dropdown.appendChild(opt);
		}
		paragraph_id = 0;
		_loadParagraph();
		dropdown.onchange = loadParagraph;
	}

	function loadButtons(){
		var field = document.getElementById("button-field");
		for(var i=0; i<10; i++){
			var b = document.createElement("button");
			b.type = "button";
			b.classList.add("btn");
			b.classList.add("btn-sm");
			if (i%2==0) b.classList.add("btn-primary");
			else b.classList.add("btn-default");
			b.id = parseInt(i);
			b.innerHTML = "Para" + parseInt(i);
			b.onclick = loadParagraph;
			field.appendChild(b);
		}
	}

	function displayQuestion(){
		var form= document.createElement("div");
		form.id = "current";
		// label for Question
		var label = document.createElement("h4");
		var span = document.createElement("span");
		span.classList.add("label");
		span.classList.add("label-primary");
		span.innerHTML="Question";
		// input for Qustion
		var input = document.createElement("input");
		input.type="text";
		input.name="question"
		input.classList.add("form-control");
		input.id="question";
		// button to submit
		var button = document.createElement("button");
		button.type = "button";
		button.classList.add("btn");
		button.classList.add("btn-sm");
		button.classList.add("btn-default");
		button.innerHTML="submit";
		button.id = "submit";
		// button to clear
		var clear = document.createElement("button");
		clear.type = "button";
		clear.classList.add("btn");
		clear.classList.add("btn-sm");
		clear.classList.add("btn-default");
		clear.innerHTML="clear";
		clear.id = "clear";
		// loading
		var loading = document.createElement("div");
		loading.id = "loading";
		loading.style.display = "none";
		var img = document.createElement("img");
		img.src = "https://webster.cs.washington.edu/images/babynames/loading.gif";
		img.alt = "icon";
		loading.appendChild(img);
		loading.innerHTML = loading.innerHTML + "loading";
		// appendChild
		form.appendChild(label);
		label.appendChild(span)
		form.appendChild(input);
		form.appendChild(button);
		form.appendChild(clear);
		form.appendChild(loading);
		var qadiv = document.getElementById("qa");
		qadiv.append(form);

		button.onclick = loadAnswer;
		clear.onclick = clearField;
	}

	function displayAnswer(answer){
	        var div = document.createElement("div");
		var label = document.createElement("h4");
		var span = document.createElement("span");
		span.classList.add("label");
		span.classList.add("label-primary");
		span.innerHTML="Answer";
		var input = document.createElement("textarea");
		input.style = "resize:none";
		input.readOnly = true;
		input.classList.add("form-control");
		input.innerHTML = answer;
		div.appendChild(label);
		label.appendChild(span);
		div.appendChild(input);

		var qadiv = document.getElementById("qa");
		qadiv.append(div);
    }

     function loadAnswer(){
		document.getElementById("loading").style.display = "block";
		var data = {
			paragraph: $("#paragraph").val(),
			question: $("#question").val()
		};
		sendAjax("/submit", data, handleAnswer);
	}

    function handleAnswer(answer){
		var curr = document.getElementById("current");
		curr.removeChild(document.getElementById("submit"));
		curr.removeChild(document.getElementById("clear"));
		curr.removeChild(document.getElementById("loading"));
		displayAnswer(answer);
		var q = document.getElementById("question");
		q.id = "";
		q.readOnly = true;

		var clear = document.createElement("button");
		clear.type = "button";
		clear.classList.add("btn");
		clear.classList.add("btn-sm");
		clear.classList.add("btn-default");
		clear.innerHTML="new question!";
		clear.id = "clear";
		clear.onclick = clearField;
		curr.appendChild(clear);	
	}

	function loadParagraph(){
		paragraph_id = this.value;
		_loadParagraph();
	}

	function _loadParagraph(){
		clearField();
		if (paragraph_id==0) p_id=1;
		document.getElementById("paragraph").value = contextss[paragraph_id];
		document.getElementById("question").value = context_questions[paragraph_id];
	}

	function handleParagraph(data){
		titles = data.titles;
		contextss = data.contextss;
                context_questions = data.context_questions;
		loadDropdown();
	}

        function sendAjax(url, data, handle){
		$.getJSON(url, data, function(response){
			handle(response.result);
		});
	}

})();
