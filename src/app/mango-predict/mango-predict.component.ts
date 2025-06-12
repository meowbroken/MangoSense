import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-mango-predict',
  standalone: false,
  templateUrl: './mango-predict.component.html',
  styleUrls: ['./mango-predict.component.css']
})
export class MangoPredictComponent {
  selectedFile: File | null = null;
  imagePreview: string | ArrayBuffer | null = null;
  prediction: any = null;

  classNames = [
   'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge',
               'Healthy', 'Powdery Mildew', 'Sooty Mould'
  ];

  constructor(private http: HttpClient) {}

  onFileSelected(event: any): void {
    this.selectedFile = event.target.files[0];
    if (this.selectedFile) {
      const reader = new FileReader();
      reader.onload = () => {
        this.imagePreview = reader.result;
      };
      reader.readAsDataURL(this.selectedFile);
    } else {
      this.imagePreview = null;
    }
  }

  onSubmit(): void {
    if (!this.selectedFile) return;

    const formData = new FormData();
    formData.append('image', this.selectedFile);

    this.http.post('http://127.0.0.1:8000/predict/', formData).subscribe({
      next: (response: any) => {

        if (typeof response.confidence === 'string' && response.confidence.endsWith('%')) {
          const num = parseFloat(response.confidence.replace('%', ''));
          response.confidence = isNaN(num) ? 0 : num / 100;
        }
        this.prediction = response;
      },
      error: (error) => {
        console.error('Prediction error:', error);
      }
    });
  }
}
