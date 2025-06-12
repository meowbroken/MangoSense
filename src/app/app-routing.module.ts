import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { MangoPredictComponent } from './mango-predict/mango-predict.component';

const routes: Routes = [
  { path: '', component: MangoPredictComponent }, // Default route
  // other routes...
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
